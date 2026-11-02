import os
import time
import json
import html
import logging
import sqlite3
import threading
from collections import deque
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import pandas as pd
import requests

# ============================================================
# НАСТРОЙКИ (БОТ ДЛЯ BTC, СПОТ С OKX)
# ============================================================

API_BASE = "https://api.coinalyze.net/v1"
COINALYZE_API_KEY = os.getenv("COINALYZE_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DRY_RUN = os.getenv("DRY_RUN", "true").strip().lower() == "true"

COIN = "BTC"
# ПЕРПЕТУАЛ БЕРЕМ С BINANCE (A), СПОТ С OKX (O)
PERP_SYMBOL = "BTCUSDT_PERP.A" 
SPOT_SYMBOL = "BTCUSDT.O" 

TIMEFRAMES = {
    "15m": (900, "15min"),
    "1h": (3600, "1hour"),
    "4h": (14400, "4hour"),
}

LOAD_ORDER = ["4h", "1h", "15m"]
HIGHER_TIMEFRAME = {"15m": "1h", "1h": "4h"}

LOOKBACK_BARS = 200 
REQUIRED_BARS = 210

CLOSE_DELAY = int(os.getenv("CLOSE_DELAY_SECONDS", "45"))
MAX_SIGNAL_DELAY = int(os.getenv("MAX_SIGNAL_DELAY_SECONDS", "600"))
MIN_SCORE = int(os.getenv("MIN_SCORE", "7"))
COOLDOWN_BARS = int(os.getenv("COOLDOWN_BARS", "2"))

MIN_DELTA_SHARE = 0.12  
OI_THRESHOLD = 0.15     
VOLUME_MULTIPLIER = 1.50
FUNDING_LIMIT = 0.03    
FUNDING_PCT_LIMIT = 0.90
MAX_DATA_LAG_BARS = 3
MAX_DATA_GAPS = 3

STATE_DB = "btc_bot_state.sqlite3"
NL = chr(10)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("btc_orderflow")

HEALTH = {"status": "starting", "last_cycle": None, "last_error": None}

# ============================================================
# HEALTH-СЕРВЕР (ДЛЯ RENDER)
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
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info("Health server: port %s", port)
    server.serve_forever()

# ============================================================
# COINALYZE CLIENT
# ============================================================
class CoinalyzeClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"api_key": COINALYZE_API_KEY, "Accept": "application/json"})
        self.calls = deque()
        self.last_call = 0.0

    def wait_for_slot(self):
        while True:
            now = time.monotonic()
            while self.calls and now - self.calls[0] >= 61:
                self.calls.popleft()
            delay = max(0.0, self.last_call + 2.1 - now)
            if len(self.calls) >= 30:
                delay = max(delay, self.calls[0] + 61 - now)
            if delay <= 0:
                self.calls.append(now)
                self.last_call = now
                return
            time.sleep(delay + 0.05)

    def get(self, endpoint, params=None):
        params = dict(params or {})
        for attempt in range(4):
            self.wait_for_slot()
            try:
                response = self.session.get(f"{API_BASE}/{endpoint}", params=params, timeout=25)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 429:
                time.sleep(60)
                continue
            if response.status_code != 200:
                raise Exception(f"{endpoint}: HTTP {response.status_code}")
            result = response.json()
            if not isinstance(result, list):
                raise Exception(f"{endpoint}: ожидался JSON-массив")
            return result
        raise Exception(f"{endpoint}: исчерпаны попытки")

    def history(self, endpoint, symbol, tf, boundary, columns, usd=False):
        seconds, interval = TIMEFRAMES[tf]
        params = {
            "symbols": symbol,
            "interval": interval,
            "from": boundary - LOOKBACK_BARS * seconds,
            "to": boundary - 1,
        }
        if usd: params["convert_to_usd"] = "true"
        result = self.get(endpoint, params)
        entry = next((item for item in result if item.get("symbol") == symbol), None)
        if entry is None or not entry.get("history"):
            raise Exception(f"{endpoint}: нет истории {symbol}")
        frame = pd.DataFrame(entry["history"])
        required = ["t"] + columns
        frame = frame[required].copy()
        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["t"] = frame["t"].astype("int64")
        return frame.drop_duplicates("t").set_index("t").sort_index()

# ============================================================
# ИНДИКАТОРЫ И ЛОГИКА
# ============================================================
def add_indicators(frame):
    frame = frame.copy()
    frame["spot_delta"] = 2 * frame["spot_buy_volume"] - frame["spot_volume"]
    frame["perp_delta"] = 2 * frame["bv"] - frame["v"]
    frame["spot_cvd"] = frame["spot_delta"].cumsum()
    frame["perp_cvd"] = frame["perp_delta"].cumsum()
    frame["spot_share"] = frame["spot_delta"] / frame["spot_volume"].replace(0, np.nan)
    frame["perp_share"] = frame["perp_delta"] / frame["v"].replace(0, np.nan)

    # Базис (Фьючерс Binance минус Спот OKX)
    frame["basis"] = (frame["c"] - frame["spot_close"]) / frame["spot_close"] * 100

    # ATR
    previous_close = frame["c"].shift(1)
    tr = pd.concat([frame["h"] - frame["l"], (frame["h"] - previous_close).abs(), (frame["l"] - previous_close).abs()], axis=1).max(axis=1)
    frame["atr"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    # EMA
    frame["ema20"] = frame["c"].ewm(span=20, adjust=False).mean()
    frame["ema200"] = frame["c"].ewm(span=200, adjust=False).mean()
    return frame

def get_context(frame):
    current = frame.iloc[-1]
    if current["c"] > current["ema200"] and current["ema20"] > current["ema200"]:
        return "bullish"
    if current["c"] < current["ema200"] and current["ema20"] < current["ema200"]:
        return "bearish"
    return "neutral"

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
        left = prices[i - order:i]
        right = prices[i + 1:i + 1 + order]
        if volumes[i] < avg_vol * 1.1: continue 

        if bullish and prices[i] < left.min() and prices[i] < right.min():
            pivots.append(i)
        elif not bullish and prices[i] > left.max() and prices[i] > right.max():
            pivots.append(i)

    if len(pivots) < 2: return False
    for k in range(len(pivots) - 1, 0, -1):
        i2 = pivots[k]
        if n - 1 - i2 > max_age: continue
        for j in range(k - 1, -1, -1):
            i1 = pivots[j]
            gap = i2 - i1
            if gap < min_gap: continue
            if gap > max_gap: break
            if bullish and prices[i2] < prices[i1] and cvd[i2] > cvd[i1]: return True
            if not bullish and prices[i2] > prices[i1] and cvd[i2] < cvd[i1]: return True
    return False

def load_market_data(client, tf, boundary):
    # Загружаем перп с Binance (A), спот с OKX (O)
    perp = client.history("ohlcv-history", PERP_SYMBOL, tf, boundary, ["o", "h", "l", "c", "v", "bv"])
    spot = client.history("ohlcv-history", SPOT_SYMBOL, tf, boundary, ["v", "bv", "c"])
    oi = client.history("open-interest-history", PERP_SYMBOL, tf, boundary, ["c"])
    funding = client.history("funding-rate-history", PERP_SYMBOL, tf, boundary, ["c"])
    liquidations = client.history("liquidation-history", PERP_SYMBOL, tf, boundary, ["l", "s"], usd=True)
    
    # Джоиним по времени (t)
    frame = perp.join(spot.rename(columns={"v": "spot_volume", "bv": "spot_buy_volume", "c": "spot_close"}), how="inner")
    frame = frame.join(oi.rename(columns={"c": "oi"}), how="inner")
    frame = frame.join(funding.rename(columns={"c": "funding"}), how="inner")
    frame = frame.join(liquidations.rename(columns={"l": "long_liq", "s": "short_liq"}), how="inner")
    
    seconds = TIMEFRAMES[tf][0]
    if len(frame) < REQUIRED_BARS:
        raise Exception(f"Недостаточно синхронизированных данных: {len(frame)} свечей")
    
    # L/S Ratio (Binance)
    frame["ls_ratio"] = np.nan
    try:
        ratio = client.history("long-short-ratio-history", PERP_SYMBOL, tf, boundary, ["r"])
        frame["ls_ratio"] = ratio["r"].where(ratio["r"] > 0).reindex(frame.index)
    except:
        pass

    return add_indicators(frame)

def analyze_orderflow(frame):
    current = frame.iloc[-1]
    previous = frame.iloc[-2]
    atr = float(frame["atr"].iloc[-2])
    if not np.isfinite(atr) or atr <= 0: return [], "Некорректный ATR"

    candle_range = current["h"] - current["l"]
    if candle_range < atr * 0.35: return [], "Слишком маленькая свеча"
    if candle_range > atr * 3: return [], "Слишком большая свеча"

    average_volume = frame["v"].iloc[-21:-1].mean()
    high_volume = current["v"] > average_volume * VOLUME_MULTIPLIER

    prior_low = frame["l"].iloc[-21:-1].min()
    prior_high = frame["h"].iloc[-21:-1].max()
    swept_low = current["l"] < prior_low and current["c"] > prior_low
    swept_high = current["h"] > prior_high and current["c"] < prior_high

    oi_change = (current["oi"] / frame["oi"].iloc[-4] - 1) * 100
    oi_buildup = oi_change > OI_THRESHOLD
    oi_flush = oi_change < -OI_THRESHOLD

    spot_flow = frame["spot_delta"].iloc[-3:].sum()
    perp_flow = frame["perp_delta"].iloc[-3:].sum()
    spot_up = current["spot_share"] > MIN_DELTA_SHARE and spot_flow > 0
    spot_down = current["spot_share"] < -MIN_DELTA_SHARE and spot_flow < 0
    perp_up = current["perp_share"] > MIN_DELTA_SHARE and perp_flow > 0
    perp_down = current["perp_share"] < -MIN_DELTA_SHARE and perp_flow < 0

    bullish_divergence = detect_divergence(frame, True)
    bearish_divergence = detect_divergence(frame, False)

    body = float(current["c"] - current["o"])
    bullish_absorption = current["perp_share"] < -MIN_DELTA_SHARE*2 and body > 0.4*atr and body > 0
    bearish_absorption = current["perp_share"] > MIN_DELTA_SHARE*2 and body < -0.4*atr and body < 0

    avg_long_liq = frame["long_liq"].iloc[-31:-1].mean()
    avg_short_liq = frame["short_liq"].iloc[-31:-1].mean()
    long_liq_spike = current["long_liq"] > 3.0 * max(avg_long_liq, 10000)
    short_liq_spike = current["short_liq"] > 3.0 * max(avg_short_liq, 10000)

    funding = float(current["funding"])
    ratio = float(current["ls_ratio"]) if pd.notna(current["ls_ratio"]) else None
    
    funding_window_bars = max(10, int(96 * 900 / TIMEFRAMES[tf][0]))
    funding_pct = float((frame["funding"].iloc[-funding_window_bars:] <= funding).mean())

    candidates = []
    rejected = []

    for direction in ["LONG", "SHORT"]:
        is_long = direction == "LONG"
        
        if is_long and not spot_up: rejected.append(f"LONG: Spot не подтверждает"); continue
        if not is_long and not spot_down: rejected.append(f"SHORT: Spot не подтверждает"); continue

        sweep = swept_low if is_long else swept_high
        divergence = bullish_divergence if is_long else bearish_divergence
        absorption = bullish_absorption if is_long else bearish_absorption
        
        if is_long:
            recovery = current["c"] > current["o"] and current["c"] > previous["c"]
            breakout = current["c"] > prior_high + 0.1 * atr
            reversal_liq = long_liq_spike
            impulse_liq = short_liq_spike
            basis_confirms = current["basis"] < 0.05 
        else:
            recovery = current["c"] < current["o"] and current["c"] < previous["c"]
            breakout = current["c"] < prior_low - 0.1 * atr
            reversal_liq = short_liq_spike
            impulse_liq = long_liq_spike
            basis_confirms = current["basis"] > 0.15 

        reversal = bool(recovery and (sweep or divergence) and (absorption or oi_flush or reversal_liq))
        impulse = bool(breakout and perp_confirms and oi_buildup and high_volume)

        if not reversal and not impulse:
            rejected.append(f"{direction}: нет сценария"); continue

        if impulse and is_long and funding > FUNDING_LIMIT and funding_pct > FUNDING_PCT_LIMIT:
            rejected.append(f"LONG: фандинг перегрет"); continue
        if impulse and not is_long and funding < -FUNDING_LIMIT and funding_pct < (1 - FUNDING_PCT_LIMIT):
            rejected.append(f"SHORT: фандинг перегрет"); continue

        setup = "Разворот" if reversal else "Пробой"
        score = 5
        reasons = [f"Spot CVD подтверждает (Share: {current['spot_share']*100:.1f}%)"]

        ls_score = 0
        if ratio is not None:
            if is_long:
                if ratio < 0.8: ls_score = 2
                elif ratio < 1.0: ls_score = 1
                elif ratio > 1.8: ls_score = -2
            else:
                if ratio > 1.5: ls_score = 2
                elif ratio > 1.2: ls_score = 1
                elif ratio < 0.7: ls_score = -2
            score += ls_score
            if ls_score != 0:
                reasons.append(f"L/S Ratio: {ratio:.2f} (Оценка: {ls_score:+})")

        if perp_confirms or (reversal and absorption): score += 1; reasons.append("Perp-поток подтверждает")
        if (impulse and oi_buildup) or (reversal and oi_flush): score += 1; reasons.append(f"OI: {oi_change:+.2f}%")
        if divergence: score += 1; reasons.append("Дивергенция CVD")
        if high_volume: score += 1; reasons.append("Объем выше среднего")
        if (reversal and reversal_liq) or (impulse and impulse_liq): score += 1; reasons.append("Всплеск ликвидаций")
        
        if reversal:
            if is_long and funding < -0.01: score += 1; reasons.append("Фандинг отрицательный (шорты платят)")
            if not is_long and funding > 0.02: score += 1; reasons.append("Фандинг положительный")

        if basis_confirms: score += 1; reasons.append(f"Базис: {current['basis']:.3f}%")

        if score < MIN_SCORE:
            rejected.append(f"{direction}: score={score} < {MIN_SCORE}"); continue

        candidates.append({
            "direction": direction, "setup": setup, "score": int(score),
            "reasons": reasons, "price": float(current["c"]), "atr": atr,
            "funding": funding, "oi_change": float(oi_change),
            "ls_ratio": ratio, "basis": float(current["basis"])
        })

    return candidates, "; ".join(rejected)

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (БД, ТЕЛЕГРАМ)
# ============================================================
class StateStore:
    def __init__(self):
        self.db = sqlite3.connect(STATE_DB)
        self.db.execute("CREATE TABLE IF NOT EXISTS signals (id TEXT PRIMARY KEY, candle INTEGER)")
        self.db.commit()

    def already_sent(self, direction, candle):
        mode = "dry" if DRY_RUN else "live"
        sid = f"BTC:{direction}:{candle}:{mode}"
        row = self.db.execute("SELECT 1 FROM signals WHERE id=?", (sid,)).fetchone()
        return row is not None

    def mark_sent(self, direction, candle):
        mode = "dry" if DRY_RUN else "live"
        sid = f"BTC:{direction}:{candle}:{mode}"
        self.db.execute("INSERT OR IGNORE INTO signals VALUES (?, ?)", (sid, candle))
        self.db.commit()

def send_telegram(text):
    if DRY_RUN:
        logger.info("DRY_RUN:\n%s", text)
        return True
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        response = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=20)
        return response.status_code == 200
    except: return False

def format_signal(tf, signal, context, higher_context):
    closed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    emoji = "🟢" if signal["direction"] == "LONG" else "🔴"
    ratio_text = f'{signal["ls_ratio"]:.2f}' if signal["ls_ratio"] else "n/a"
    
    lines = [
        f"{emoji} <b>BTC | {tf.upper()} {signal['direction']}</b>",
        f"Цена: {signal['price']:,.2f}",
        f"Время (UTC): {closed_at}",
        f"Сетап: {signal['setup']}",
        f"Баллы: {signal['score']}/12",
        "",
        f"📊 L/S Ratio: {ratio_text}",
        f"⚖️ Базис (Binance/OKX): {signal['basis']:.3f}%",
        f"🏦 OI за 3 свечи: {signal['oi_change']:+.2f}%",
        f"💰 Funding: {signal['funding']:+.4f}%",
        "",
        f"Контекст 1h: {higher_context}",
        f"Контекст 4h: {context}",
        "",
        "Причины:"
    ]
    lines.extend("• " + html.escape(r) for r in signal["reasons"])
    return NL.join(lines)

# ============================================================
# ГЛАВНЫЙ ЦИКЛ
# ============================================================
def get_boundaries():
    effective_time = int(time.time()) - CLOSE_DELAY
    return {tf: (effective_time // settings[0]) * settings[0] for tf, settings in TIMEFRAMES.items()}

def main():
    if not COINALYZE_API_KEY: raise RuntimeError("Нет API ключа")
    threading.Thread(target=start_health_server, daemon=True).start()
    
    client = CoinalyzeClient()
    state = StateStore()
    cache = {}
    data_retry_after = {}

    send_telegram("🚀 <b>BTC Order Flow Bot запущен</b>\nСпот: OKX | Перп: Binance\nРежим: " + ("DRY RUN" if DRY_RUN else "LIVE"))
    HEALTH["status"] = "running"

    while True:
        boundaries = get_boundaries()
        cycle_errors = False

        for tf in LOAD_ORDER:
            boundary = boundaries[tf]
            existing = cache.get(tf)
            if existing is not None and existing["boundary"] == boundary: continue
            if time.time() < data_retry_after.get(tf, 0): cycle_errors = True; continue
            
            try:
                frame = load_market_data(client, tf, boundary)
                cache[tf] = {"boundary": boundary, "frame": frame, "context": get_context(frame)}
                data_retry_after.pop(tf, None)
                logger.info(f"DATA OK | BTC {tf} | bars={len(frame)} | ctx={cache[tf]['context']}")
            except Exception as exc:
                cycle_errors = True
                cache.pop(tf, None)
                data_retry_after[tf] = time.time() + 120
                logger.error(f"Data error {tf}: {exc}")

        tf = "15m"
        record = cache.get(tf)
        if record and record["boundary"] == boundaries[tf]:
            frame = record["frame"]
            candle = int(frame.index[-1])
            
            if not state.already_sent("ANY", candle): 
                candidates, explanation = analyze_orderflow(frame)
                
                if len(candidates) == 1:
                    signal = candidates[0]
                    dir = signal["direction"]
                    
                    if not state.already_sent(dir, candle):
                        h1_ctx = cache.get("1h", {}).get("context", "neutral")
                        h4_ctx = cache.get("4h", {}).get("context", "neutral")
                        
                        msg = format_signal(tf, signal, h4_ctx, h1_ctx)
                        if send_telegram(msg):
                            state.mark_sent(dir, candle)
                            state.mark_sent("ANY", candle) 
                            logger.info(f"Signal sent: {dir} | Score: {signal['score']}")

        HEALTH["last_cycle"] = int(time.time())
        if not cycle_errors: HEALTH["last_error"] = None
        time.sleep(15)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopped")
