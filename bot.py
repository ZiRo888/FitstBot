import os
import time
import json
import html
import logging
import sqlite3
import threading
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import pandas as pd
import requests

# ============================================================
# НАСТРОЙКИ (1H + 4H, БЕЗ EMA200, С АЛЕРТАМИ ПО ЛИКВИДАЦИЯМ)
# ============================================================

API_BASE = "https://api.coinalyze.net/v1"
COINALYZE_API_KEY = os.getenv("COINALYZE_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DRY_RUN = os.getenv("DRY_RUN", "false").strip().lower() == "true"

TIMEZONE_NAME = os.getenv("TIMEZONE", "Europe/Moscow").strip()
try:
    LOCAL_TIMEZONE = ZoneInfo(TIMEZONE_NAME)
except (ZoneInfoNotFoundError, ValueError):
    LOCAL_TIMEZONE = ZoneInfo("UTC")

coins = [value.strip().upper() for value in os.getenv("WATCHLIST", "BTC").split(",") if value.strip()]
WATCHLIST = list(dict.fromkeys(["BTC"] + coins))

TIMEFRAMES = {
    "1h": (3600, "1hour"),
    "4h": (14400, "4hour"),
}

LOAD_ORDER = ["4h", "1h"]
LOOKBACK_BARS = 150 
REQUIRED_BARS = 100

CLOSE_DELAY = int(os.getenv("CLOSE_DELAY_SECONDS", "45"))
MAX_SIGNAL_DELAY_BY_TF = {"1h": 3600, "4h": 14400}

MIN_SCORE = int(os.getenv("MIN_SCORE", "7")) 
COOLDOWN_BARS = int(os.getenv("COOLDOWN_BARS", "2"))

MIN_DELTA_SHARE = float(os.getenv("MIN_DELTA_SHARE", "0.08"))
OI_THRESHOLD = float(os.getenv("OI_CHANGE_PERCENT", "0.20"))
VOLUME_MULTIPLIER = float(os.getenv("VOLUME_MULTIPLIER", "1.50"))
FUNDING_LIMIT = float(os.getenv("FUNDING_LIMIT_PERCENT", "0.04"))
FUNDING_PCT_LIMIT = float(os.getenv("FUNDING_PCT_LIMIT", "0.90"))

DATA_RETRY_SECONDS = 120
SPOT_RECHECK_SECONDS = 900
STATE_DB = "bot_state.sqlite3"
NL = chr(10)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("orderflow")
HEALTH = {"status": "starting", "last_cycle": None, "last_error": None}

# ============================================================
# ИНФРАСТРУКТУРА
# ============================================================
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(HEALTH, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args): pass

def start_health_server():
    port = int(os.getenv("PORT", "10000"))
    HTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()

def self_ping_worker():
    url = os.getenv("RENDER_EXTERNAL_URL", "")
    if not url: return
    while True:
        try: requests.get(url, timeout=10)
        except Exception: pass
        time.sleep(600)

class DataError(Exception): pass
class AuthenticationError(Exception): pass

class CoinalyzeClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"api_key": COINALYZE_API_KEY, "Accept": "application/json"})
        self.calls = deque()
        self.last_call = 0.0
        self.rate_lock = threading.Lock()

    def wait_for_slot(self):
        with self.rate_lock:
            while True:
                now = time.monotonic()
                while self.calls and now - self.calls[0] >= 61.0:
                    self.calls.popleft()
                delay = max(0.0, self.last_call + 2.1 - now)
                if len(self.calls) >= 30:
                    delay = max(delay, self.calls[0] + 61.0 - now)
                if delay <= 0:
                    self.calls.append(now); self.last_call = now; return
                time.sleep(delay + 0.05)

    def get(self, endpoint, params=None):
        params = dict(params or {})
        for attempt in range(4):
            self.wait_for_slot()
            try:
                res = self.session.get(f"{API_BASE}/{endpoint}", params=params, timeout=25)
            except requests.RequestException:
                time.sleep(2 ** attempt); continue
            if res.status_code == 401: raise AuthenticationError("API Key Error")
            if res.status_code == 429: time.sleep(60); continue
            if res.status_code != 200: raise DataError(f"{endpoint}: HTTP {res.status_code}")
            return res.json()
        raise DataError(f"{endpoint}: failed")

    def history(self, endpoint, symbol, tf, boundary, columns, usd=False):
        seconds, interval = TIMEFRAMES[tf]
        params = {"symbols": symbol, "interval": interval, "from": boundary - LOOKBACK_BARS * seconds, "to": boundary - 1}
        if usd: params["convert_to_usd"] = "true"
        
        result = self.get(endpoint, params)
        entry = next((item for item in result if item.get("symbol") == symbol), None)
        if entry is None or not entry.get("history"): raise DataError(f"{endpoint}: no history {symbol}")
        
        frame = pd.DataFrame(entry["history"])[["t"] + columns]
        for col in ["t"] + columns: frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame.drop_duplicates("t").set_index("t").sort_index()

# ============================================================
# ПОИСК РЫНКОВ
# ============================================================
def find_spot_market(client, coin, tf, boundary):
    spots = client.get("spot-markets")
    candidates = []
    for ex in ["A", "O", "B", "C"]: 
        c = [m for m in spots if m.get("exchange") == ex and m.get("base_asset") == coin and m.get("quote_asset") == "USDT"]
        candidates.extend(sorted(c, key=lambda x: x["symbol"]))
    for cand in candidates:
        try:
            probe = client.history("ohlcv-history", cand["symbol"], tf, boundary, ["v", "bv"])
            if len(probe) >= REQUIRED_BARS: return cand
        except: pass
    return None

def build_market_map(client):
    futures = client.get("future-markets")
    mapping = {}
    boundaries = {tf: (int(time.time()) - CLOSE_DELAY) // spec[0] * spec[0] for tf, spec in TIMEFRAMES.items()}

    for coin in WATCHLIST:
        perps = [m for m in futures if m.get("exchange") == "A" and m.get("base_asset") == coin and m.get("quote_asset") == "USDT" and m.get("is_perpetual")]
        if not perps: continue
        
        perp = sorted(perps, key=lambda x: x["symbol"])[0]
        spot_by_tf = {}
        for tf in TIMEFRAMES:
            spot = find_spot_market(client, coin, tf, boundaries[tf])
            if spot: spot_by_tf[tf] = spot
            
        if spot_by_tf:
            mapping[coin] = {"perp": perp, "spot_by_tf": spot_by_tf}
            logger.info(f"MARKET OK | {coin} | Perp: {perp['symbol']}")
            
    return mapping

# ============================================================
# ЛОГИКА АНАЛИЗА
# ============================================================
def add_indicators(frame):
    frame = frame.copy()
    frame["spot_delta"] = 2 * frame["spot_buy_volume"] - frame["spot_volume"]
    frame["perp_delta"] = 2 * frame["bv"] - frame["v"]
    frame["spot_cvd"] = frame["spot_delta"].cumsum()
    frame["perp_cvd"] = frame["perp_delta"].cumsum()
    frame["spot_share"] = frame["spot_delta"] / frame["spot_volume"].replace(0, np.nan)
    frame["perp_share"] = frame["perp_delta"] / frame["v"].replace(0, np.nan)
    
    if "spot_close" in frame.columns:
        frame["basis"] = (frame["c"] - frame["spot_close"]) / frame["spot_close"] * 100

    previous_close = frame["c"].shift(1)
    tr = pd.concat([frame["h"] - frame["l"], (frame["h"] - previous_close).abs(), (frame["l"] - previous_close).abs()], axis=1).max(axis=1)
    frame["atr"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return frame

def detect_divergence(frame, bullish, lookback=40, order=2, min_gap=3, max_gap=15, max_age=12):
    if len(frame) < lookback: return False
    window = frame.iloc[-lookback:]
    prices = (window["l"] if bullish else window["h"]).to_numpy()
    cvd = window["spot_cvd"].to_numpy()
    volumes = window["v"].to_numpy()
    avg_vol = volumes.mean()
    n = len(prices)
    
    pivots = []
    for i in range(order, n - order):
        left = prices[i - order:i]; right = prices[i + 1:i + 1 + order]
        if volumes[i] < avg_vol * 1.1: continue 
        if bullish and prices[i] < left.min() and prices[i] < right.min(): pivots.append(i)
        elif not bullish and prices[i] > left.max() and prices[i] > right.max(): pivots.append(i)

    if len(pivots) < 2: return False
    for k in range(len(pivots) - 1, 0, -1):
        i2 = pivots[k]
        if n - 1 - i2 > max_age: continue
        for j in range(k - 1, -1, -1):
            i1 = pivots[j]; gap = i2 - i1
            if gap < min_gap: continue
            if gap > max_gap: break
            if bullish and prices[i2] < prices[i1] and cvd[i2] > cvd[i1]: return True
            if not bullish and prices[i2] > prices[i1] and cvd[i2] < cvd[i1]: return True
    return False

def load_market_data(client, market, tf, boundary, coin="?"):
    spot_m = market["spot_by_tf"].get(tf)
    if not spot_m: raise DataError(f"No spot {coin} {tf}")
    
    perp = client.history("ohlcv-history", market["perp"]["symbol"], tf, boundary, ["o", "h", "l", "c", "v", "bv"])
    spot = client.history("ohlcv-history", spot_m["symbol"], tf, boundary, ["v", "bv", "c"])
    oi = client.history("open-interest-history", market["perp"]["symbol"], tf, boundary, ["c"])
    funding = client.history("funding-rate-history", market["perp"]["symbol"], tf, boundary, ["c"])
    
    try: liq = client.history("liquidation-history", market["perp"]["symbol"], tf, boundary, ["l", "s"], usd=True)
    except: liq = pd.DataFrame(columns=["l", "s"], index=pd.Index([], name="t"))

    try: ls = client.history("long-short-ratio-history", market["perp"]["symbol"], tf, boundary, ["r"])
    except: ls = pd.DataFrame(columns=["r"], index=pd.Index([], name="t"))

    frame = perp.join(spot.rename(columns={"v": "spot_volume", "bv": "spot_buy_volume", "c": "spot_close"}), how="inner")
    frame = frame.join(oi.rename(columns={"c": "oi"}), how="inner")
    frame = frame.join(funding.rename(columns={"c": "funding"}), how="inner")
    
    if not liq.empty: frame = frame.join(liq.rename(columns={"l": "long_liq", "s": "short_liq"}), how="left")
    else: frame["long_liq"] = np.nan; frame["short_liq"] = np.nan
        
    if not ls.empty: frame["ls_ratio"] = ls["r"].where(ls["r"] > 0).reindex(frame.index)
    else: frame["ls_ratio"] = np.nan

    if len(frame) < REQUIRED_BARS: raise DataError(f"Not enough bars: {len(frame)}")
    return add_indicators(frame)

def analyze_orderflow(frame):
    current = frame.iloc[-1]; prev = frame.iloc[-2]
    atr = current["atr"]
    if not np.isfinite(atr) or atr <= 0: return [], "Bad ATR", None

    c_range = current["h"] - current["l"]
    if c_range < atr * 0.35 or c_range > atr * 3: return [], "Bad candle size", None

    avg_vol = frame["v"].iloc[-21:-1].mean()
    high_vol = current["v"] > avg_vol * VOLUME_MULTIPLIER

    prior_low = frame["l"].iloc[-21:-1].min()
    prior_high = frame["h"].iloc[-21:-1].max()
    swept_low = current["l"] < prior_low and current["c"] > prior_low
    swept_high = current["h"] > prior_high and current["c"] < prior_high

    oi_change = (current["oi"] / frame["oi"].iloc[-4] - 1) * 100
    oi_buildup = oi_change > OI_THRESHOLD
    oi_flush = oi_change < -OI_THRESHOLD

    spot_up = current["spot_share"] > MIN_DELTA_SHARE and frame["spot_delta"].iloc[-3:].sum() > 0
    spot_down = current["spot_share"] < -MIN_DELTA_SHARE and frame["spot_delta"].iloc[-3:].sum() < 0
    perp_up = current["perp_share"] > MIN_DELTA_SHARE and frame["perp_delta"].iloc[-3:].sum() > 0
    perp_down = current["perp_share"] < -MIN_DELTA_SHARE and frame["perp_delta"].iloc[-3:].sum() < 0

    bull_div = detect_divergence(frame, True)
    bear_div = detect_divergence(frame, False)
    body = current["c"] - current["o"]

    bull_abs = current["perp_share"] < -MIN_DELTA_SHARE*2 and body > 0.4*atr and body > 0
    bear_abs = current["perp_share"] > MIN_DELTA_SHARE*2 and body < -0.4*atr and body < 0

    # === ЛОГИКА ЛИКВИДАЦИЙ (ТРИГГЕР) ===
    avg_long_liq = frame["long_liq"].iloc[-31:-1].mean() if "long_liq" in frame else 0
    avg_short_liq = frame["short_liq"].iloc[-31:-1].mean() if "short_liq" in frame else 0
    
    long_liq_spike = current.get("long_liq", 0) > 3.0 * max(avg_long_liq, 10000)
    short_liq_spike = current.get("short_liq", 0) > 3.0 * max(avg_short_liq, 10000)

    # Формируем Алерт для Telegram
    liq_alert = None
    if long_liq_spike:
        liq_alert = "⚠️ <b>ВНИМАНИЕ: Вынос ЛОНГОВ</b>"
    elif short_liq_spike:
        liq_alert = "⚠️ <b>ВНИМАНИЕ: Вынос ШОРТОВ</b>"

    funding = float(current["funding"])
    ratio = float(current["ls_ratio"]) if pd.notna(current["ls_ratio"]) else None
    basis = float(current.get("basis", 0))

    candidates = []
    
    for direction in ["LONG", "SHORT"]:
        is_long = direction == "LONG"
        if is_long and not spot_up: continue
        if not is_long and not spot_down: continue

        sweep = swept_low if is_long else swept_high
        div = bull_div if is_long else bear_div
        absorp = bull_abs if is_long else bear_abs
        
        if is_long:
            rec = current["c"] > current["o"] and current["c"] > prev["c"]
            breakout = current["c"] > prior_high + 0.1 * atr
            rev_liq = long_liq_spike; imp_liq = short_liq_spike
            trap_trigger = short_liq_spike
            trap_confirm = spot_up and rec
        else:
            rec = current["c"] < current["o"] and current["c"] < prev["c"]
            breakout = current["c"] < prior_low - 0.1 * atr
            rev_liq = short_liq_spike; imp_liq = long_liq_spike
            trap_trigger = long_liq_spike
            trap_confirm = spot_down and rec

        reversal = bool(rec and (sweep or div) and (absorp or oi_flush or rev_liq))
        impulse = bool(breakout and (perp_up if is_long else perp_down) and oi_buildup and high_vol)
        
        # Новая логика: Ловушка после ликвидаций
        liquidation_trap = bool(trap_trigger and trap_confirm and (perp_up if is_long else perp_down))

        if not reversal and not impulse and not liquidation_trap: continue

        score = 5
        reasons = [f"Spot OK ({current['spot_share']*100:.1f}%)"]
        
        if liquidation_trap:
            score += 2
            reasons.append(f"Триггер: {'Вынос шортов' if is_long else 'Вынос лонгов'} + Поток")
            
        if div: 
            score += 2 
            reasons.append("Дивергенция CVD")
            
        if (reversal and rev_liq) or (impulse and imp_liq): 
            score += 2 
            reasons.append("Всплеск ликвидаций")

        if ratio is not None:
            if is_long:
                if ratio < 0.8: score += 1; reasons.append(f"L/S {ratio:.2f} (Сквиз)")
                elif ratio > 1.8: score -= 1
            else:
                if ratio > 1.5: score += 1; reasons.append(f"L/S {ratio:.2f} (Сквиз)")
                elif ratio < 0.7: score -= 1
        
        if (perp_up if is_long else perp_down) or (reversal and absorp): score += 1; reasons.append("Perp OK")
        if (impulse and oi_buildup) or (reversal and oi_flush): score += 1; reasons.append(f"OI {oi_change:+.2f}%")
        if high_vol: score += 1; reasons.append("Объем")
        
        if reversal and is_long and funding < -0.01: score += 1; reasons.append("Funding минус")
        if reversal and not is_long and funding > 0.02: score += 1; reasons.append("Funding плюс")

        if score >= MIN_SCORE:
            setup_name = "Ловушка" if liquidation_trap else ("Разворот" if reversal else "Пробой")
            candidates.append({
                "direction": direction, 
                "setup": setup_name, 
                "score": score, 
                "reasons": reasons, 
                "price": current["c"], 
                "funding": funding, 
                "ls": ratio, 
                "basis": basis, 
                "oi": oi_change
            })

    return candidates, "", liq_alert

# ============================================================
# ТЕЛЕГРАМ И БД
# ============================================================
class StateStore:
    def __init__(self):
        self.db = sqlite3.connect(STATE_DB)
        self.db.execute("CREATE TABLE IF NOT EXISTS signals (id TEXT PRIMARY KEY)")
        self.db.commit()
    def sent(self, c, tf, d, cd): return self.db.execute("SELECT 1 FROM signals WHERE id=?", (f"{c}:{tf}:{d}:{cd}",)).fetchone() is not None
    def mark(self, c, tf, d, cd):
        self.db.execute("INSERT OR IGNORE INTO signals VALUES (?)", (f"{c}:{tf}:{d}:{cd}",))
        self.db.commit()

def send_telegram(text):
    if DRY_RUN: logger.info(f"DRY_RUN:\n{text}"); return True
    try:
        return requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=20).status_code == 200
    except: return False

def format_signal(coin, tf, candle, sig):
    t = (candle + TIMEFRAMES[tf][0])
    time_str = datetime.fromtimestamp(t, LOCAL_TIMEZONE).strftime("%d.%m %H:%M")
    emoji = "🟢" if sig["direction"] == "LONG" else "🔴"
    ls_str = f'{sig["ls"]:.2f}' if sig["ls"] else "n/a"
    
    msg = f"{emoji} <b>{coin} | {tf.upper()} {sig['direction']}</b>\n"
    msg += f"Цена: {sig['price']:,.2f} | Время: {time_str}\n"
    msg += f"Сетап: {sig['setup']} (Баллы: {sig['score']})\n\n"
    msg += f"📊 L/S: {ls_str} | Базис: {sig['basis']:.3f}%\n"
    msg += f"🏦 OI: {sig['oi']:+.2f}% | Funding: {sig['funding']:+.4f}%\n\n"
    msg += "Причины:\n" + "\n".join([f"• {r}" for r in sig["reasons"]])
    return msg

# ============================================================
# ЗАПУСК
# ============================================================
def main():
    threading.Thread(target=start_health_server, daemon=True).start()
    threading.Thread(target=self_ping_worker, daemon=True).start()
    
    client = CoinalyzeClient()
    state = StateStore()
    cache, processed = {}, set()

    send_telegram("🚀 <b>Order Flow Bot (v2.1) запущен</b>\nСистема Алертов (с датой и временем) активирована.")
    market_map = build_market_map(client)
    if not market_map: raise Exception("Нет монет для торговли")

    while True:
        now = int(time.time())
        boundaries = {tf: (now - CLOSE_DELAY) // spec[0] * spec[0] for tf, spec in TIMEFRAMES.items()}
        
        # Загрузка данных
        for tf in LOAD_ORDER:
            for coin in market_map:
                if cache.get((coin, tf), {}).get("boundary") == boundaries[tf]: continue
                try:
                    frame = load_market_data(client, market_map[coin], tf, boundaries[tf], coin)
                    cache[(coin, tf)] = {"boundary": boundaries[tf], "frame": frame}
                    logger.info(f"DATA OK | {coin} {tf}")
                except Exception as e:
                    logger.warning(f"DATA ERR | {coin} {tf} | {e}")

        # Анализ
        for tf in LOAD_ORDER:
            for coin in market_map:
                rec = cache.get((coin, tf))
                if not rec or rec["boundary"] != boundaries[tf]: continue
                
                frame = rec["frame"]
                candle = int(frame.index[-1])
                
                if (coin, tf, candle) in processed: continue
                if time.time() - (candle + TIMEFRAMES[tf][0]) > MAX_SIGNAL_DELAY_BY_TF[tf]:
                    processed.add((coin, tf, candle)); continue

                # Получаем сигналы и алерт
                candidates, _, liq_alert = analyze_orderflow(frame)

                # --- ОТПРАВЛЯЕМ АЛЕРТ (С датой и временем) ---
                alert_key = f"{coin}:{tf}:{candle}:alert"
                if liq_alert and alert_key not in processed:
                    # Считаем время закрытия свечи (когда именно была палка)
                    candle_close_time = candle + TIMEFRAMES[tf][0]
                    # Переводим в твой часовой пояс
                    time_str = datetime.fromtimestamp(candle_close_time, LOCAL_TIMEZONE).strftime('%d.%m.%Y %H:%M')
                    
                    # Считаем, сколько минут прошло с момента закрытия свечи
                    age_minutes = int((time.time() - candle_close_time) / 60)
                    
                    # Если сигнал пришел с задержкой более 15 минут, добавляем предупреждение
                    age_text = f"\n⚠️ <i>Задержка: {age_minutes} мин. (старая свеча)</i>" if age_minutes > 15 else ""

                    alert_msg = (
                        f"{liq_alert}\n"
                        f"Монета: <b>{coin}</b> | ТФ: <b>{tf.upper()}</b>\n"
                        f"📅 Время свечи: <b>{time_str}</b>{age_text}\n"
                        f"Начинаю следить за потоком..."
                    )
                    send_telegram(alert_msg)
                    logger.info(f"ALERT SENT | {coin} {tf} | {time_str}")
                    processed.add(alert_key)

                # --- ОТПРАВЛЯЕМ СИГНАЛ (Если набрались баллы) ---
                if len(candidates) == 1:
                    sig = candidates[0]
                    if not state.sent(coin, tf, sig["direction"], candle):
                        if send_telegram(format_signal(coin, tf, candle, sig)):
                            state.mark(coin, tf, sig["direction"], candle)
                            logger.info(f"SIGNAL SENT | {coin} {tf} {sig['direction']} | Score: {sig['score']}")
                
                processed.add((coin, tf, candle))

        time.sleep(15)

if __name__ == "__main__":
    main()
