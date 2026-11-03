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
# НАСТРОЙКИ
# ============================================================

API_BASE = "https://api.coinalyze.net/v1"
COINALYZE_API_KEY = os.getenv("COINALYZE_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

DRY_RUN = os.getenv("DRY_RUN", "true").strip().lower() == "true"

coins = [
    value.strip().upper()
    for value in os.getenv("WATCHLIST", "BTC,SOL,XRP,BNB,DOGE,LINK").split(",")
    if value.strip()
]

WATCHLIST = list(dict.fromkeys(["BTC"] + coins))

TIMEFRAMES = {
    "1h": (3600, "1hour"),
    "4h": (14400, "4hour"),
}
LOAD_ORDER = ["4h", "1h"]
HIGHER_TIMEFRAME = {"1h": "4h"}

LOOKBACK_BARS = 100
REQUIRED_BARS = 65

CLOSE_DELAY = int(os.getenv("CLOSE_DELAY_SECONDS", "45"))
# Максимальный возраст сигнала при отправке: отдельно для каждого таймфрейма.
# Значение можно переопределить через Render Environment.
MAX_SIGNAL_DELAY_BY_TF = {
    "1h": int(os.getenv("MAX_SIGNAL_DELAY_1H_SECONDS", "3600")),
    "4h": int(os.getenv("MAX_SIGNAL_DELAY_4H_SECONDS", "14400")),
}
MIN_SCORE = int(os.getenv("MIN_SCORE", "10"))
COOLDOWN_BARS = int(os.getenv("COOLDOWN_BARS", "2"))

MIN_DELTA_SHARE = float(os.getenv("MIN_DELTA_SHARE", "0.03"))
OI_THRESHOLD = float(os.getenv("OI_CHANGE_PERCENT", "0.30"))
VOLUME_MULTIPLIER = float(os.getenv("VOLUME_MULTIPLIER", "1.50"))
FUNDING_LIMIT = float(os.getenv("FUNDING_LIMIT_PERCENT", "0.05"))
FUNDING_PCT_LIMIT = float(os.getenv("FUNDING_PCT_LIMIT", "0.90"))
FUNDING_WINDOW = int(os.getenv("FUNDING_WINDOW_BARS", "96"))
MIN_LIQ_USD = float(os.getenv("MIN_LIQUIDATION_USD", "10000"))
LIQ_MULTIPLIER = float(os.getenv("LIQUIDATION_SPIKE_MULTIPLIER", "3.0"))

DATA_RETRY_SECONDS = 120
TELEGRAM_RETRY_SECONDS = 60
STATE_DB = os.getenv("STATE_DB", "bot_state.sqlite3")
REQUIRED_EXCHANGE = "A" # Binance
SPOT_FALLBACK_EXCHANGES = ["A", "O", "B", "C"]

NL = chr(10)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("orderflow")

HEALTH = {"status": "starting", "last_cycle": None, "last_data_success": None, "last_error": None}

# ============================================================
# HEALTH-СЕРВЕР & ПИНГЕР
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

def self_ping_worker():
    url = os.getenv("RENDER_EXTERNAL_URL", "https://fitstbot.onrender.com/")
    if not url.endswith("/"): url += "/"
    while True:
        try:
            res = requests.get(url, timeout=10)
            logger.info("Self-ping OK | HTTP %s", res.status_code)
        except Exception as exc:
            logger.warning("Self-ping failed: %s", type(exc).__name__)
        time.sleep(600)

# ============================================================
# COINALYZE CLIENT
# ============================================================

class DataError(Exception): pass
class AuthenticationError(Exception): pass

class CoinalyzeClient:
    # Оставляем запас относительно лимита Coinalyze (40 запросов/мин).
    # Учитываем каждый HTTP-запрос, включая повторные попытки после ошибок.
    MAX_CALLS_PER_WINDOW = 30
    WINDOW_SECONDS = 61.0
    MIN_REQUEST_GAP = 2.1

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"api_key": COINALYZE_API_KEY, "Accept": "application/json"})
        self.calls = deque()
        self.last_call = 0.0
        self.total_calls = 0
        self.rate_lock = threading.Lock()

    def wait_for_slot(self):
        # Один общий диспетчер для всех монет, таймфреймов и повторных запросов.
        # Lock не позволяет двум потокам одновременно занять один слот.
        with self.rate_lock:
            while True:
                now = time.monotonic()
                while self.calls and now - self.calls[0] >= self.WINDOW_SECONDS:
                    self.calls.popleft()
                delay = max(0.0, self.last_call + self.MIN_REQUEST_GAP - now)
                if len(self.calls) >= self.MAX_CALLS_PER_WINDOW:
                    delay = max(delay, self.calls[0] + self.WINDOW_SECONDS - now)
                if delay <= 0:
                    self.calls.append(now)
                    self.last_call = now
                    self.total_calls += 1
                    return
                time.sleep(delay + 0.05)

    def get(self, endpoint, params=None):
        params = dict(params or {})
        for attempt in range(4):
            self.wait_for_slot()
            try:
                response = self.session.get(f"{API_BASE}/{endpoint}", params=params, timeout=25)
            except requests.RequestException:
                time.sleep(2 ** attempt); continue

            if response.status_code == 401: raise AuthenticationError("Проверьте COINALYZE_API_KEY")
            if response.status_code == 429:
                delay = float(response.headers.get("Retry-After", "60"))
                logger.warning("Coinalyze 429 | Сплю %.1f секунд", delay)
                time.sleep(max(1, delay)); continue
            if response.status_code >= 500:
                time.sleep(2 ** attempt); continue
            if response.status_code != 200: raise DataError(f"{endpoint}: HTTP {response.status_code}")

            try: result = response.json()
            except ValueError: time.sleep(2 ** attempt); continue
            if not isinstance(result, list): raise DataError(f"{endpoint}: ожидался JSON-массив")
            return result
        raise DataError(f"{endpoint}: исчерпаны попытки")

    def history(self, endpoint, symbol, tf, boundary, columns, usd=False):
        seconds, interval = TIMEFRAMES[tf]
        params = {"symbols": symbol, "interval": interval, "from": boundary - LOOKBACK_BARS * seconds, "to": boundary - 1}
        if usd: params["convert_to_usd"] = "true"
        result = self.get(endpoint, params)
        entry = next((item for item in result if item.get("symbol") == symbol), None)
        if entry is None or not entry.get("history"): raise DataError(f"{endpoint}: нет истории {symbol}")

        frame = pd.DataFrame(entry["history"])
        required = ["t"] + columns
        if not set(required).issubset(frame.columns): raise DataError(f"{endpoint}: нет полей {required}")
        frame = frame[required].copy()
        for column in required: frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["t"] = frame["t"].astype("int64")
        frame = frame.drop_duplicates("t").set_index("t").sort_index()
        return frame.loc[frame.index + seconds <= boundary]


def _is_required_exchange(market_exchange):
    value = str(market_exchange or "").strip()
    return value == REQUIRED_EXCHANGE or (REQUIRED_EXCHANGE == "A" and value == "Binance")

def exchange_name(code):
    mapping = {"A": "Binance", "B": "Bybit", "C": "Coinbase", "O": "OKX"}
    return mapping.get(str(code or "").strip(), str(code or "?"))

def build_market_map(client):
    spots = client.get("spot-markets")
    futures = client.get("future-markets")
    mapping = {}

    # Проверяем глубину Spot-истории отдельно для каждого TF. Это позволяет,
    # например, оставить XRP на 4H, даже если для 1H нет достаточной истории.
    now = int(time.time())
    probe_boundaries = {
        tf: ((now - CLOSE_DELAY) // spec[0]) * spec[0]
        for tf, spec in TIMEFRAMES.items()
    }

    for coin in WATCHLIST:
        perps = [m for m in futures if _is_required_exchange(m.get("exchange")) and m.get("base_asset") == coin and m.get("quote_asset") == "USDT" and bool(m.get("is_perpetual")) and bool(m.get("has_ohlcv_data"))]
        if not perps:
            logger.warning("MARKET SKIP | %s | Perp=False | монета исключена", coin)
            continue

        perps.sort(key=lambda m: (not bool(m.get("has_long_short_ratio_data")), m["symbol"]))
        spot_by_tf = {}
        for tf in TIMEFRAMES:
            selected = None
            best_count = -1
            candidates_for_coin = []
            for ex_code in SPOT_FALLBACK_EXCHANGES:
                exchange_candidates = [
                    m for m in spots
                    if str(m.get("exchange", "")).strip() == ex_code
                    and m.get("base_asset") == coin
                    and m.get("quote_asset") == "USDT"
                    and "USDT" in str(m.get("symbol", "")).upper()
                ]
                candidates_for_coin.extend(sorted(exchange_candidates, key=lambda m: m["symbol"]))

            for candidate in candidates_for_coin:
                symbol = candidate["symbol"]
                try:
                    probe = client.history(
                        "ohlcv-history", symbol, tf, probe_boundaries[tf], ["v", "bv"]
                    )
                    bars = len(probe)
                    best_count = max(best_count, bars)
                    if bars >= REQUIRED_BARS:
                        selected = candidate
                        logger.info(
                            "SPOT SELECT | %s %s | %s (%s) | bars=%d",
                            coin, tf, symbol, exchange_name(candidate.get("exchange")), bars
                        )
                        break
                    logger.warning(
                        "SPOT REJECT | %s %s | %s (%s) | bars=%d < %d",
                        coin, tf, symbol, exchange_name(candidate.get("exchange")), bars, REQUIRED_BARS
                    )
                except Exception as exc:
                    logger.warning(
                        "SPOT PROBE ERR | %s %s | %s (%s) | %s",
                        coin, tf, symbol, exchange_name(candidate.get("exchange")), exc
                    )

            spot_by_tf[tf] = selected
            if selected is None:
                logger.warning(
                    "SPOT UNAVAILABLE | %s %s | нет Spot-рынка с историей >= %d баров (лучшая выборка=%d)",
                    coin, tf, REQUIRED_BARS, best_count
                )

        if not any(spot_by_tf.values()):
            logger.warning("MARKET SKIP | %s | нет Spot-истории достаточной длины ни на одном TF", coin)
            continue

        mapping[coin] = {"spot_by_tf": spot_by_tf, "perp": perps[0]}
        selected_text = ", ".join(
            f"{tf}={market['symbol'] if market else 'нет'}"
            for tf, market in spot_by_tf.items()
        )
        logger.info("MARKET OK | %s | spot[%s] | perp=%s", coin, selected_text, perps[0]["symbol"])

    if not mapping:
        raise DataError("Не удалось найти ни одной монеты с подходящими Spot и perpetual рынками")
    skipped = [coin for coin in WATCHLIST if coin not in mapping]
    logger.info("MARKET SUMMARY | active=%s | skipped=%s", ",".join(mapping), ",".join(skipped) or "нет")
    return mapping

# ============================================================
# ИНДИКАТОРЫ И АНАЛИЗ
# ============================================================

def add_indicators(frame):
    frame = frame.copy()
    frame["spot_delta"] = 2 * frame["spot_buy_volume"] - frame["spot_volume"]
    frame["perp_delta"] = 2 * frame["bv"] - frame["v"]
    frame["spot_cvd"] = frame["spot_delta"].cumsum()
    frame["perp_cvd"] = frame["perp_delta"].cumsum()
    frame["spot_share"] = frame["spot_delta"] / frame["spot_volume"].replace(0, np.nan)
    frame["perp_share"] = frame["perp_delta"] / frame["v"].replace(0, np.nan)

    previous_close = frame["c"].shift(1)
    tr = pd.concat([frame["h"] - frame["l"], (frame["h"] - previous_close).abs(), (frame["l"] - previous_close).abs()], axis=1).max(axis=1)
    frame["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    frame["ema20"] = frame["c"].ewm(span=20, adjust=False).mean()
    return frame

def rolling_pct(series, window):
    min_periods = max(10, window // 4)
    return series.rolling(window, min_periods=min_periods).apply(lambda x: float((x <= x[-1]).mean()) if len(x) > 0 else 0.5, raw=True)

def get_context(frame):
    current = frame.iloc[-1]
    spot_flow = frame["spot_delta"].iloc[-3:].sum()
    perp_flow = frame["perp_delta"].iloc[-3:].sum()
    if current["c"] > current["ema20"] and spot_flow > 0 and perp_flow > 0: return "bullish"
    if current["c"] < current["ema20"] and spot_flow < 0 and perp_flow < 0: return "bearish"
    return "neutral"

def detect_divergence(frame, bullish, lookback=40, order=2, min_gap=3, max_gap=15, max_age=12):
    if len(frame) < lookback: return False
    window = frame.iloc[-lookback:]
    prices = (window["l"] if bullish else window["h"]).to_numpy()
    cvd = window["spot_cvd"].to_numpy()
    n = len(prices)
    pivots = []
    for i in range(order, n - order):
        left, right = prices[i - order:i], prices[i + 1:i + 1 + order]
        if bullish:
            if prices[i] < left.min() and prices[i] < right.min(): pivots.append(i)
        else:
            if prices[i] > left.max() and prices[i] > right.max(): pivots.append(i)
    if len(pivots) < 2: return False
    for k in range(len(pivots) - 1, 0, -1):
        i2 = pivots[k]
        if n - 1 - i2 > max_age: continue
        for j in range(k - 1, -1, -1):
            i1 = pivots[j]
            gap = i2 - i1
            if gap < min_gap: continue
            if gap > max_gap: break
            if bullish:
                if prices[i2] < prices[i1] and cvd[i2] > cvd[i1]: return True
            else:
                if prices[i2] > prices[i1] and cvd[i2] < cvd[i1]: return True
    return False

def analyze_orderflow(frame):
    current, previous = frame.iloc[-1], frame.iloc[-2]
    atr = float(frame["atr"].iloc[-2])
    if not np.isfinite(atr) or atr <= 0: return [], "Некорректный ATR"
    candle_range = current["h"] - current["l"]
    if candle_range < atr * 0.35: return [], "Маленькая свеча"
    if candle_range > atr * 3: return [], "Растянутая свеча"
    average_volume = frame["v"].iloc[-21:-1].mean()
    if average_volume <= 0 or current["v"] <= 0 or current["spot_volume"] <= 0: return [], "Мало объёма"

    high_volume = current["v"] > average_volume * VOLUME_MULTIPLIER
    prior_low, prior_high = frame["l"].iloc[-21:-1].min(), frame["h"].iloc[-21:-1].max()
    swept_low = current["l"] < prior_low and current["c"] > prior_low
    swept_high = current["h"] > prior_high and current["c"] < prior_high
    bullish_fvg = current["l"] > frame["h"].iloc[-3]
    bearish_fvg = current["h"] < frame["l"].iloc[-3]

    oi_change = (current["oi"] / frame["oi"].iloc[-4] - 1) * 100
    oi_buildup, oi_flush = oi_change > OI_THRESHOLD, oi_change < -OI_THRESHOLD

    spot_flow, perp_flow = frame["spot_delta"].iloc[-3:].sum(), frame["perp_delta"].iloc[-3:].sum()
    spot_up = current["spot_share"] > MIN_DELTA_SHARE and spot_flow > 0
    spot_down = current["spot_share"] < -MIN_DELTA_SHARE and spot_flow < 0
    perp_up = current["perp_share"] > MIN_DELTA_SHARE and perp_flow > 0
    perp_down = current["perp_share"] < -MIN_DELTA_SHARE and perp_flow < 0

    bullish_div, bearish_div = detect_divergence(frame, True), detect_divergence(frame, False)
    body = float(current["c"] - current["o"])
    bull_abs = current["perp_share"] < -MIN_DELTA_SHARE * 2 and current["v"] > average_volume * 1.3 and abs(body) > 0.4 * atr and body > 0
    bear_abs = current["perp_share"] > MIN_DELTA_SHARE * 2 and current["v"] > average_volume * 1.3 and abs(body) > 0.4 * atr and body < 0

    funding = float(current["funding"])
    funding_pct = float(rolling_pct(frame["funding"], FUNDING_WINDOW).iloc[-1])

    candidates, rejected = [], []
    for direction in ["LONG", "SHORT"]:
        is_long = direction == "LONG"
        spot_conf = spot_up if is_long else spot_down
        perp_conf = perp_up if is_long else perp_down
        sweep = swept_low if is_long else swept_high
        div = bullish_div if is_long else bearish_div
        absorp = bull_abs if is_long else bear_abs

        if not spot_conf:
            rejected.append(f"{direction}: Spot нет")
            continue

        if is_long:
            recovery = current["c"] > current["o"] and current["c"] > previous["c"]
            breakout = current["c"] > prior_high + 0.1 * atr
        else:
            recovery = current["c"] < current["o"] and current["c"] < previous["c"]
            breakout = current["c"] < prior_low - 0.1 * atr

        reversal = bool(recovery and (sweep or div) and (absorp or oi_flush))
        impulse = bool(breakout and perp_conf and oi_buildup and high_volume)

        if not reversal and not impulse:
            rejected.append(f"{direction}: нет сценария")
            continue

        if impulse and is_long and funding > FUNDING_LIMIT and funding_pct > FUNDING_PCT_LIMIT:
            rejected.append(f"{direction}: funding"); continue
        if impulse and not is_long and funding < -FUNDING_LIMIT and funding_pct < (1 - FUNDING_PCT_LIMIT):
            rejected.append(f"{direction}: funding"); continue

        setup = "Разворот" if reversal else "Пробой"
        score = 4
        reasons = ["Spot: дельта", ("Разворот" if reversal else "Пробой")]
        if perp_conf or (reversal and absorp): score += 1; reasons.append("Perp")
        if (impulse and oi_buildup) or (reversal and oi_flush): score += 1; reasons.append("OI")
        if div: score += 1; reasons.append("Дивергенция")
        if high_volume: score += 1; reasons.append("Объём")
        
        ratio = float(current["ls_ratio"]) if pd.notna(current.get("ls_ratio")) else None
        if ratio is not None and ((is_long and ratio < 0.8) or (not is_long and ratio > 1.25)):
            score += 1; reasons.append("L/S")

        if score < MIN_SCORE:
            rejected.append(f"{direction}: score={score}")
            continue

        candidates.append({
            "direction": direction, "setup": setup, "score": int(score), "reasons": reasons,
            "price": float(current["c"]), "atr": atr, "oi_change": float(oi_change),
            "funding": funding, "funding_pct": funding_pct,
            "spot_delta_pct": float(current["spot_share"] * 100), "perp_delta_pct": float(current["perp_share"] * 100),
            "long_liq": 0.0, "short_liq": 0.0, "ls_ratio": ratio, "fvg": bool(bullish_fvg if is_long else bearish_fvg),
        })
    return candidates, "; ".join(rejected)

# ============================================================
# ОСНОВНОЙ ЦИКЛ
# ============================================================

class StateStore:
    def __init__(self):
        self.db = sqlite3.connect(STATE_DB)
        self.db.execute("CREATE TABLE IF NOT EXISTS of_signals (id TEXT PRIMARY KEY, coin TEXT, timeframe TEXT, direction TEXT, candle INTEGER, mode TEXT)")
        self.db.commit()

    def already_sent(self, coin, tf, direction, candle):
        mode = "dry" if DRY_RUN else "live"
        sid = f"{coin}:{tf}:{direction}:{candle}:{mode}"
        return self.db.execute("SELECT 1 FROM of_signals WHERE id=?", (sid,)).fetchone() is not None

    def mark_sent(self, coin, tf, direction, candle):
        mode = "dry" if DRY_RUN else "live"
        sid = f"{coin}:{tf}:{direction}:{candle}:{mode}"
        self.db.execute("INSERT OR IGNORE INTO of_signals VALUES (?, ?, ?, ?, ?, ?)", (sid, coin, tf, direction, int(candle), mode))
        self.db.commit()

def load_market_data(client, market, tf, boundary, coin="?"):
    spot_market = market.get("spot_by_tf", {}).get(tf)
    if spot_market is None:
        raise DataError(f"для {coin} {tf} нет Spot-рынка с историей >= {REQUIRED_BARS} баров")
    perp_sym, spot_sym = market["perp"]["symbol"], spot_market["symbol"]

    # Получаем источники отдельно и сразу записываем размер каждой выборки.
    perp = client.history("ohlcv-history", perp_sym, tf, boundary, ["o", "h", "l", "c", "v", "bv"])
    logger.info("DATA PART | %s %s | Perp=%d", coin, tf, len(perp))

    spot = client.history("ohlcv-history", spot_sym, tf, boundary, ["v", "bv"])
    logger.info("DATA PART | %s %s | Spot=%d", coin, tf, len(spot))

    oi = client.history("open-interest-history", perp_sym, tf, boundary, ["c"])
    logger.info("DATA PART | %s %s | OI=%d", coin, tf, len(oi))

    funding = client.history("funding-rate-history", perp_sym, tf, boundary, ["c"])
    logger.info("DATA PART | %s %s | Funding=%d", coin, tf, len(funding))

    frame = perp.join(spot.rename(columns={"v": "spot_volume", "bv": "spot_buy_volume"}), how="inner")
    count_after_spot = len(frame)
    frame = frame.join(oi.rename(columns={"c": "oi"}), how="inner")
    count_after_oi = len(frame)
    frame = frame.join(funding.rename(columns={"c": "funding"}), how="inner")
    logger.info(
        "DATA JOIN | %s %s | Perp=%d Spot=%d OI=%d Funding=%d | after_spot=%d after_oi=%d final=%d",
        coin, tf, len(perp), len(spot), len(oi), len(funding),
        count_after_spot, count_after_oi, len(frame)
    )

    frame["long_liq"], frame["short_liq"] = 0.0, 0.0
    frame["ls_ratio"] = np.nan

    if len(frame) < REQUIRED_BARS:
        raise DataError(
            f"Мало баров после inner join: {len(frame)} < {REQUIRED_BARS} "
            f"(Perp={len(perp)}, Spot={len(spot)}, OI={len(oi)}, Funding={len(funding)})"
        )
    return add_indicators(frame)

def send_telegram(text):
    if DRY_RUN:
        logger.info("DRY_RUN: %s", text)
        return True
    try:
        res = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=20)
        return res.status_code == 200
    except Exception: return False

def format_signal(coin, tf, candle, signal, detected_at):
    signal_time = candle + TIMEFRAMES[tf][0]
    age_seconds = max(0, int(detected_at) - signal_time)
    signal_time_text = datetime.fromtimestamp(signal_time, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    detected_text = datetime.fromtimestamp(detected_at, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    age_minutes = age_seconds // 60
    emoji = "🟢" if signal["direction"] == "LONG" else "🔴"
    return NL.join([
        f"{emoji} <b>ORDER FLOW | {tf.upper()} {signal['direction']}</b>", "",
        f"Монета: <b>#{coin}USDT</b>", f"Цена сигнала: <b>{signal['price']:,.4f}</b>",
        f"Время сигнала (закрытие свечи): {signal_time_text}",
        f"Бот обнаружил/отправляет: {detected_text}",
        f"Возраст сигнала: {age_minutes} мин.",
        f"Сетап: {signal['setup']}", f"Баллы: {signal['score']}/10", "",
        f"📊 Spot: {signal['spot_delta_pct']:+.1f}%", f"📊 Perp: {signal['perp_delta_pct']:+.1f}%",
        f"🏦 OI: {signal['oi_change']:+.2f}%", f"💰 Funding: {signal['funding']:+.4f}%"
    ])

def main():
    threading.Thread(target=start_health_server, daemon=True).start()
    threading.Thread(target=self_ping_worker, daemon=True).start()

    client = CoinalyzeClient()
    state = StateStore()
    cache, processed = {}, set()
    retry_after = {}  # (coin, timeframe, boundary) -> unix timestamp следующей попытки

    # В текущей загрузке L/S Ratio не запрашивается, поэтому фактически доступны 8 баллов.
    logger.info(
        "CONFIG | REQUESTED_WATCHLIST=%s | TIMEFRAMES=%s | MIN_SCORE=%d | REQUIRED_BARS=%d | MAX_SIGNAL_DELAY_1H=%d sec | MAX_SIGNAL_DELAY_4H=%d sec | DRY_RUN=%s",
        ",".join(WATCHLIST), ",".join(LOAD_ORDER), MIN_SCORE, REQUIRED_BARS,
        MAX_SIGNAL_DELAY_BY_TF["1h"], MAX_SIGNAL_DELAY_BY_TF["4h"], DRY_RUN
    )
    if MIN_SCORE > 8:
        logger.warning(
            "CONFIG WARNING | MIN_SCORE=%d выше текущего максимума 8: сигналы не пройдут score-фильтр "
            "(L/S Ratio пока не загружается)", MIN_SCORE
        )

    send_telegram(f"🚀 <b>Order Flow (Упрощенный) запущен</b>\nМонеты: {', '.join(WATCHLIST)}\nРежим: {'DRY RUN' if DRY_RUN else 'LIVE'}")
    HEALTH["status"] = "starting"
    market_map = build_market_map(client)
    active_watchlist = list(market_map)
    logger.info(
        "SCHEDULER | активных монет=%d | таймфреймов=%d | запросов на полное обновление=%d | лимит=%d запросов/%d сек",
        len(active_watchlist), len(TIMEFRAMES),
        len(active_watchlist) * len(TIMEFRAMES) * 4,
        client.MAX_CALLS_PER_WINDOW, int(client.WINDOW_SECONDS)
    )
    if len(active_watchlist) < 3:
        logger.warning("WATCHLIST WARNING | доступно менее 3 монет: %s", ",".join(active_watchlist))
    HEALTH["status"] = "running"

    while True:
        now = int(time.time())
        boundaries = {tf: ((now - CLOSE_DELAY) // spec[0]) * spec[0] for tf, spec in TIMEFRAMES.items()}

        for tf in LOAD_ORDER:
            for coin in active_watchlist:
                if market_map[coin].get("spot_by_tf", {}).get(tf) is None:
                    continue
                key = (coin, tf)
                boundary = boundaries[tf]
                if cache.get(key) and cache[key]["boundary"] == boundary:
                    continue

                retry_key = (coin, tf, boundary)
                retry_time = retry_after.get(retry_key, 0)
                if now < retry_time:
                    continue

                try:
                    frame = load_market_data(client, market_map[coin], tf, boundary, coin)
                    cache[key] = {"boundary": boundary, "frame": frame}
                    retry_after.pop(retry_key, None)
                    HEALTH["last_data_success"] = int(time.time())
                    HEALTH["last_error"] = None
                    HEALTH["status"] = "running"
                    logger.info("DATA OK | %s %s | bars=%s", coin, tf, len(frame))
                except Exception as exc:
                    retry_after[retry_key] = int(time.time()) + DATA_RETRY_SECONDS
                    HEALTH["last_error"] = f"{coin} {tf}: {type(exc).__name__}: {exc}"
                    HEALTH["status"] = "degraded"
                    logger.warning(
                        "DATA ERR | %s %s | %s | повтор после %d сек",
                        coin, tf, exc, DATA_RETRY_SECONDS
                    )

        for tf in LOAD_ORDER:
            for coin in active_watchlist:
                if market_map[coin].get("spot_by_tf", {}).get(tf) is None:
                    continue
                rec = cache.get((coin, tf))
                if not rec or rec["boundary"] != boundaries[tf]:
                    continue
                frame = rec["frame"]
                candle = int(frame.index[-1])
                work_key = (coin, tf, candle)
                if work_key in processed:
                    continue

                # Не отправляем исторические сигналы при старте или после задержки API.
                # candle — время открытия свечи; прибавляем длительность TF, чтобы получить
                # фактическое время её закрытия.
                candle_close_time = candle + TIMEFRAMES[tf][0]
                max_signal_age = MAX_SIGNAL_DELAY_BY_TF[tf]
                signal_age = int(time.time()) - candle_close_time
                if signal_age < 0 or signal_age > max_signal_age:
                    logger.info(
                        "SIGNAL SKIP STALE | %s %s | age=%d sec | max=%d sec | candle_close=%s",
                        coin, tf, signal_age, max_signal_age,
                        datetime.fromtimestamp(candle_close_time, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    )
                    processed.add(work_key)
                    continue

                candidates, explanation = analyze_orderflow(frame)
                if not candidates:
                    logger.info("%s %s | skip: %s", coin, tf, explanation)
                    processed.add(work_key)
                    continue

                signal = candidates[0]
                if state.already_sent(coin, tf, signal["direction"], candle):
                    processed.add(work_key)
                    continue

                # Повторная проверка непосредственно перед отправкой: анализ/очередь
                # могли занять время, и сигнал мог выйти за возрастной лимит TF.
                detected_at = int(time.time())
                signal_age = detected_at - candle_close_time
                if signal_age < 0 or signal_age > max_signal_age:
                    logger.info(
                        "SIGNAL SKIP STALE BEFORE SEND | %s %s | age=%d sec | max=%d sec",
                        coin, tf, signal_age, max_signal_age
                    )
                    processed.add(work_key)
                    continue

                message = format_signal(coin, tf, candle, signal, detected_at)
                if send_telegram(message):
                    state.mark_sent(coin, tf, signal["direction"], candle)
                    logger.info("SIGNAL: %s %s %s | age=%d sec", coin, tf, signal["direction"], signal_age)
                    processed.add(work_key)

        HEALTH["last_cycle"] = int(time.time())
        time.sleep(30)

if __name__ == "__main__":
    main()
