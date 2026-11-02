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
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests


# ============================================================
# НАСТРОЙКИ
# ============================================================

BINANCE_SPOT_API = "https://api.binance.com/api/v3"
BINANCE_FUT_API = "https://fapi.binance.com/fapi/v1"
BINANCE_FUT_DATA_API = "https://fapi.binance.com/futures/data"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

DRY_RUN = os.getenv("DRY_RUN", "true").strip().lower() == "true"

coins = [
    value.strip().upper()
    for value in os.getenv("WATCHLIST", "BTC,ETH,SOL").split(",")
    if value.strip()
]

WATCHLIST = list(dict.fromkeys(["BTC"] + coins))

TIMEFRAMES = {
    "15m": (900, "15m"),
    "1h": (3600, "1h"),
    "4h": (14400, "4h"),
}

LOAD_ORDER = ["4h", "1h", "15m"]
HIGHER_TIMEFRAME = {"15m": "1h", "1h": "4h"}

LOOKBACK_BARS = 100
REQUIRED_BARS = 65

CLOSE_DELAY = int(os.getenv("CLOSE_DELAY_SECONDS", "45"))
MAX_SIGNAL_DELAY = int(os.getenv("MAX_SIGNAL_DELAY_SECONDS", "600"))
MIN_SCORE = int(os.getenv("MIN_SCORE", "7"))
COOLDOWN_BARS = int(os.getenv("COOLDOWN_BARS", "2"))

MIN_DELTA_SHARE = float(os.getenv("MIN_DELTA_SHARE", "0.03"))
OI_THRESHOLD = float(os.getenv("OI_CHANGE_PERCENT", "0.30"))
VOLUME_MULTIPLIER = float(os.getenv("VOLUME_MULTIPLIER", "1.50"))

FUNDING_LIMIT = float(os.getenv("FUNDING_LIMIT_PERCENT", "0.05"))
FUNDING_PCT_LIMIT = float(os.getenv("FUNDING_PCT_LIMIT", "0.90"))
FUNDING_WINDOW = int(os.getenv("FUNDING_WINDOW_BARS", "96"))

MIN_LIQ_USD = float(os.getenv("MIN_LIQUIDATION_USD", "10000"))
LIQ_MULTIPLIER = float(os.getenv("LIQUIDATION_SPIKE_MULTIPLIER", "3.0"))

DATA_RETRY_SECONDS = 60
TELEGRAM_RETRY_SECONDS = 60

STATE_DB = os.getenv("STATE_DB", "bot_state.sqlite3")

# --- Бэктест ---
RUN_BACKTEST = os.getenv("RUN_BACKTEST", "false").strip().lower() == "true"
BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "30"))  # Максимум 30 из-за OI!
BACKTEST_TP_ATR = float(os.getenv("BACKTEST_TP_ATR", "1.5"))
BACKTEST_SL_ATR = float(os.getenv("BACKTEST_SL_ATR", "1.0"))
BACKTEST_MAX_BARS = int(os.getenv("BACKTEST_MAX_BARS", "30"))
BACKTEST_TFS = [
    t.strip() for t in os.getenv("BACKTEST_TFS", "1h,4h").split(",")
    if t.strip()
]
BACKTEST_MIN_SCORE = int(
    os.getenv("BACKTEST_MIN_SCORE", os.getenv("MIN_SCORE", "5"))
)

NL = chr(10)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("orderflow")

HEALTH = {
    "status": "starting",
    "last_cycle": None,
    "last_data_success": None,
    "last_error": None,
    "watchlist": WATCHLIST,
}


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

    def log_message(self, *args):
        pass


def start_health_server():
    port = int(os.getenv("PORT", "10000"))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info("Health server: port %s", port)
    server.serve_forever()


def self_ping_worker():
    """Пингует свой же URL каждые 10 минут, чтобы Render не усыплял сервис."""
    url = os.getenv("RENDER_EXTERNAL_URL", "https://fitstbot.onrender.com/")
    if not url.endswith("/"):
        url += "/"
    
    while True:
        try:
            res = requests.get(url, timeout=10)
            logger.info("Self-ping OK | HTTP %s", res.status_code)
        except Exception as exc:
            logger.warning("Self-ping failed: %s", type(exc).__name__)
        time.sleep(600)


# ============================================================
# BINANCE API CLIENT
# ============================================================

class DataError(Exception):
    pass

class BinanceClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get(self, url, params=None):
        for attempt in range(4):
            try:
                response = self.session.get(url, params=params, timeout=15)
            except requests.RequestException as exc:
                logger.warning("Network error | %s | %s", url, type(exc).__name__)
                time.sleep(2 ** attempt)
                continue

            if response.status_code == 429 or response.status_code == 418:
                logger.warning("Binance rate limit. Sleeping 60 sec.")
                time.sleep(60)
                continue
                
            if response.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
                
            if response.status_code != 200:
                raise DataError(f"HTTP {response.status_code} | {url}")

            try:
                result = response.json()
            except ValueError:
                time.sleep(2 ** attempt)
                continue

            return result

        raise DataError(f"{url}: exhausted retries")


def exchange_symbol(coin):
    return f"{coin}USDT"

def fetch_binance_klines(client, is_futures, symbol, tf, start_ms, end_ms):
    url = f"{BINANCE_FUT_API}/klines" if is_futures else f"{BINANCE_SPOT_API}/klines"
    params = {
        "symbol": symbol,
        "interval": TIMEFRAMES[tf][1],
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000
    }
    res = client.get(url, params)
    
    data = []
    for k in res:
        t = int(k[0]) // 1000
        o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
        bv = float(k[9])
        data.append([t, o, h, l, c, v, bv])
        
    df = pd.DataFrame(data, columns=["t", "o", "h", "l", "c", "v", "bv"])
    return df.set_index("t")


def load_market_data(client, coin, tf, boundary):
    seconds = TIMEFRAMES[tf][0]
    end_ms = boundary * 1000
    start_ms = (boundary - LOOKBACK_BARS * seconds) * 1000
    sym = exchange_symbol(coin)

    try:
        # Свечи перпетуала
        perp = fetch_binance_klines(client, True, sym, tf, start_ms, end_ms)
        
        # Свечи спота
        spot = fetch_binance_klines(client, False, sym, tf, start_ms, end_ms)
        if spot.empty: raise DataError("Спот пуст")
        
        # Открытый интерес (OI)
        oi_res = client.get(f"{BINANCE_FUT_DATA_API}/openInterestHist", {
            "symbol": sym, "period": TIMEFRAMES[tf][1], "limit": 500
        })
        oi_df = pd.DataFrame(oi_res)
        if not oi_df.empty:
            oi_df["t"] = oi_df["timestamp"].astype(int) // 1000
            oi_df["oi"] = oi_df["sumOpenInterest"].astype(float)
            oi_df = oi_df.set_index("t")[["oi"]]
        else:
            raise DataError("Нет данных OI")

        # Фандинг (берем последнюю известную ставку)
        fund_res = client.get(f"{BINANCE_FUT_API}/fundingRate", {"symbol": sym, "limit": 1})
        funding_val = float(fund_res[0]["fundingRate"]) * 100 if fund_res else 0.0

        # L/S Ratio
        ls_res = client.get(f"{BINANCE_FUT_DATA_API}/globalLongShortAccountRatio", {
            "symbol": sym, "period": TIMEFRAMES[tf][1], "limit": 500
        })
        ls_df = pd.DataFrame(ls_res)
        if not ls_df.empty:
            ls_df["t"] = ls_df["timestamp"].astype(int) // 1000
            ls_df["ls_ratio"] = ls_df["longShortRatio"].astype(float)
            ls_df = ls_df.set_index("t")[["ls_ratio"]]
        else:
            ls_df = pd.DataFrame(index=perp.index)

        # Собираем всё вместе
        frame = perp.rename(columns={"v": "v", "bv": "bv"})
        frame = frame.join(spot.rename(columns={"v": "spot_volume", "bv": "spot_buy_volume"}), how="inner")
        frame = frame.join(oi_df, how="left")
        frame = frame.join(ls_df, how="left")
        
        # Заполняем фандинг и ликвидации (заглушка)
        frame["funding"] = funding_val
        frame["long_liq"] = 0.0
        frame["short_liq"] = 0.0

        # Очистка
        frame = frame.ffill().bfill().dropna()
        
        if len(frame) < REQUIRED_BARS:
            raise DataError(f"Мало баров: {len(frame)}")

        # Оставляем только завершенные свечи
        frame = frame.loc[frame.index + seconds <= boundary]
        
        return add_indicators(frame)

    except Exception as e:
        raise DataError(str(e))


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
    tr = pd.concat([
        frame["h"] - frame["l"],
        (frame["h"] - previous_close).abs(),
        (frame["l"] - previous_close).abs(),
    ], axis=1).max(axis=1)

    frame["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    frame["ema20"] = frame["c"].ewm(span=20, adjust=False).mean()

    return frame


def rolling_pct(series, window):
    min_periods = max(10, window // 4)
    return series.rolling(window, min_periods=min_periods).apply(
        lambda x: float((x <= x[-1]).mean()) if len(x) > 0 else 0.5,
        raw=True,
    )


def get_context(frame):
    current = frame.iloc[-1]
    spot_flow = frame["spot_delta"].iloc[-3:].sum()
    perp_flow = frame["perp_delta"].iloc[-3:].sum()

    if current["c"] > current["ema20"] and spot_flow > 0 and perp_flow > 0:
        return "bullish"
    if current["c"] < current["ema20"] and spot_flow < 0 and perp_flow < 0:
        return "bearish"
    return "neutral"


def detect_divergence(frame, bullish, lookback=40, order=2, min_gap=3, max_gap=15, max_age=12):
    if len(frame) < lookback:
        return False

    window = frame.iloc[-lookback:]
    prices = (window["l"] if bullish else window["h"]).to_numpy()
    cvd = window["spot_cvd"].to_numpy()
    n = len(prices)

    pivots = []
    for i in range(order, n - order):
        left = prices[i - order:i]
        right = prices[i + 1:i + 1 + order]
        if bullish:
            if prices[i] < left.min() and prices[i] < right.min(): pivots.append(i)
        else:
            if prices[i] > left.max() and prices[i] > right.max(): pivots.append(i)

    if len(pivots) < 2:
        return False

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
    current = frame.iloc[-1]
    previous = frame.iloc[-2]
    atr = float(frame["atr"].iloc[-2])

    if not np.isfinite(atr) or atr <= 0: return [], "Некорректный ATR"
    candle_range = current["h"] - current["l"]
    if candle_range < atr * 0.35: return [], "Слишком маленькая свеча по ATR"
    if candle_range > atr * 3: return [], "Слишком растянутая свеча по ATR"

    average_volume = frame["v"].iloc[-21:-1].mean()
    if average_volume <= 0 or current["v"] <= 0 or current["spot_volume"] <= 0:
        return [], "Недостаточный объём"

    high_volume = current["v"] > average_volume * VOLUME_MULTIPLIER
    prior_low = frame["l"].iloc[-21:-1].min()
    prior_high = frame["h"].iloc[-21:-1].max()

    swept_low = current["l"] < prior_low and current["c"] > prior_low
    swept_high = current["h"] > prior_high and current["c"] < prior_high
    bullish_fvg = current["l"] > frame["h"].iloc[-3]
    bearish_fvg = current["h"] < frame["l"].iloc[-3]

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
    body_abs = abs(body)

    bullish_absorption = (
        current["perp_share"] < -MIN_DELTA_SHARE * 2 and current["v"] > average_volume * 1.3
        and body_abs > 0.4 * atr and body > 0
    )
    bearish_absorption = (
        current["perp_share"] > MIN_DELTA_SHARE * 2 and current["v"] > average_volume * 1.3
        and body_abs > 0.4 * atr and body < 0
    )

    average_long_liq = frame["long_liq"].iloc[-31:-1].mean()
    average_short_liq = frame["short_liq"].iloc[-31:-1].mean()

    long_liq_spike = (current["long_liq"] >= MIN_LIQ_USD and current["long_liq"] > LIQ_MULTIPLIER * max(average_long_liq, 1.0))
    short_liq_spike = (current["short_liq"] >= MIN_LIQ_USD and current["short_liq"] > LIQ_MULTIPLIER * max(average_short_liq, 1.0))

    funding = float(current["funding"])
    funding_pct_series = rolling_pct(frame["funding"], FUNDING_WINDOW)
    funding_pct = float(funding_pct_series.iloc[-1]) if pd.notna(funding_pct_series.iloc[-1]) else 0.5

    candidates = []
    rejected = []

    for direction in ["LONG", "SHORT"]:
        is_long = direction == "LONG"
        spot_confirms = spot_up if is_long else spot_down
        perp_confirms = perp_up if is_long else perp_down
        sweep = swept_low if is_long else swept_high
        divergence = bullish_divergence if is_long else bearish_divergence
        absorption = bullish_absorption if is_long else bearish_absorption

        if not spot_confirms:
            rejected.append(f"{direction}: Spot не подтверждает")
            continue

        if is_long:
            recovery = current["c"] > current["o"] and current["c"] > previous["c"]
            breakout = current["c"] > prior_high + 0.1 * atr
            reversal_liq = long_liq_spike
            impulse_liq = short_liq_spike
        else:
            recovery = current["c"] < current["o"] and current["c"] < previous["c"]
            breakout = current["c"] < prior_low - 0.1 * atr
            reversal_liq = short_liq_spike
            impulse_liq = long_liq_spike

        reversal = bool(recovery and (sweep or divergence) and (absorption or oi_flush or reversal_liq))
        impulse = bool(breakout and perp_confirms and oi_buildup and high_volume)

        if not reversal and not impulse:
            rejected.append(f"{direction}: нет ценового сценария")
            continue

        if impulse and is_long and funding > FUNDING_LIMIT and funding_pct > FUNDING_PCT_LIMIT:
            rejected.append(f"{direction}: funding перегрет")
            continue
        if impulse and not is_long and funding < -FUNDING_LIMIT and funding_pct < (1 - FUNDING_PCT_LIMIT):
            rejected.append(f"{direction}: funding перегрет")
            continue

        setup = "Разворот после свипа/дивергенции" if reversal else "Подтверждённый пробой"
        score = 4
        reasons = ["Spot: дельта и поток за 3 свечи подтверждают", ("Цена подтвердила разворот у уровня" if reversal else "Пробой подтверждён CVD Perp, OI и объёмом")]

        if perp_confirms or (reversal and absorption):
            score += 1
            reasons.append("Perp-поток подтверждает" if perp_confirms else "Возможное поглощение")
        if (impulse and oi_buildup) or (reversal and oi_flush):
            score += 1
            reasons.append("OI: рост открытых позиций" if oi_buildup else "OI: сокращение открытых позиций")
        if divergence:
            score += 1
            reasons.append("Дивергенция цены и Spot CVD")
        if high_volume:
            score += 1
            reasons.append("Объём выше среднего")
        if reversal_liq if reversal else impulse_liq:
            score += 1
            reasons.append("Ликвидации аномальны")

        ratio = float(current["ls_ratio"]) if pd.notna(current.get("ls_ratio")) else None
        if ratio is not None and ((is_long and ratio < 0.8) or (not is_long and ratio > 1.25)):
            score += 1
            reasons.append("L/S: контрнаправленный перекос")

        if score < MIN_SCORE:
            rejected.append(f"{direction}: score={score}, нужно {MIN_SCORE}")
            continue

        candidates.append({
            "direction": direction, "setup": setup, "score": int(score), "reasons": reasons,
            "price": float(current["c"]), "atr": atr, "oi_change": float(oi_change),
            "funding": funding, "funding_pct": funding_pct,
            "spot_delta_pct": float(current["spot_share"] * 100), "perp_delta_pct": float(current["perp_share"] * 100),
            "long_liq": float(current["long_liq"]), "short_liq": float(current["short_liq"]),
            "ls_ratio": ratio, "fvg": bool(bullish_fvg if is_long else bearish_fvg),
        })

    return candidates, "; ".join(rejected)


# ============================================================
# БЭКТЕСТ
# ============================================================

def load_backtest_frame(client, coin, tf, days):
    seconds = TIMEFRAMES[tf][0]
    end_t = (int(time.time()) // seconds) * seconds
    start_t = end_t - days * 86400
    start_ms, end_ms = start_t * 1000, end_t * 1000
    
    sym = exchange_symbol(coin)
    perp = fetch_binance_klines(client, True, sym, tf, start_ms, end_ms)
    spot = fetch_binance_klines(client, False, sym, tf, start_ms, end_ms)
    
    oi_res = client.get(f"{BINANCE_FUT_DATA_API}/openInterestHist", {"symbol": sym, "period": TIMEFRAMES[tf][1], "limit": 500})
    oi_df = pd.DataFrame(oi_res)
    if not oi_df.empty:
        oi_df["t"] = oi_df["timestamp"].astype(int) // 1000
        oi_df["oi"] = oi_df["sumOpenInterest"].astype(float)
        oi_df = oi_df.set_index("t")[["oi"]]
    
    frame = perp.join(spot.rename(columns={"v": "spot_volume", "bv": "spot_buy_volume"}), how="inner")
    frame = frame.join(oi_df, how="left").ffill().bfill()
    frame["funding"] = 0.0
    frame["long_liq"] = 0.0
    frame["short_liq"] = 0.0
    
    return add_indicators(frame)


def simulate_trade(frame, idx, direction, tp_atr, sl_atr, max_bars):
    entry = float(frame.iloc[idx]["c"])
    atr = float(frame["atr"].iloc[idx - 1])
    if not np.isfinite(atr) or atr <= 0: return "invalid", 0.0, 0

    if direction == "LONG":
        tp, sl = entry + tp_atr * atr, entry - sl_atr * atr
    else:
        tp, sl = entry - tp_atr * atr, entry + sl_atr * atr

    end = min(idx + 1 + max_bars, len(frame))
    for j in range(idx + 1, end):
        bar = frame.iloc[j]
        if direction == "LONG":
            if bar["l"] <= sl: return "SL", -1.0, j - idx
            if bar["h"] >= tp: return "TP", tp_atr / sl_atr, j - idx
        else:
            if bar["h"] >= sl: return "SL", -1.0, j - idx
            if bar["l"] <= tp: return "TP", tp_atr / sl_atr, j - idx

    last_j = min(idx + max_bars, len(frame) - 1)
    exit_px = float(frame.iloc[last_j]["c"])
    pnl_r = (exit_px - entry) / (sl_atr * atr) if direction == "LONG" else (entry - exit_px) / (sl_atr * atr)
    return "TIMEOUT", pnl_r, last_j - idx

def _summary(subset):
    if not subset: return None
    pnls = np.array([t["pnl_r"] for t in subset], dtype=float)
    wins = pnls > 0
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity) if len(equity) else np.array([0.0])
    dd = float((equity - peak).min()) if len(equity) else 0.0
    gains = pnls[wins].sum() if wins.any() else 0.0
    losses = pnls[~wins].sum() if (~wins).any() else 0.0
    pf = float(gains / abs(losses)) if losses < 0 else float("inf")
    return {"n": int(len(pnls)), "wr": float(wins.mean() * 100), "avg_r": float(pnls.mean()), "total_r": float(pnls.sum()), "pf": pf, "max_dd": dd}

def _fmt(label, s):
    if s is None: return f"  {label}: НЕТ СДЕЛОК"
    return f"  {label}: N={s['n']} WR={s['wr']:.1f}% AvgR={s['avg_r']:+.3f} TotalR={s['total_r']:+.2f} PF={s['pf']:.2f} MaxDD={s['max_dd']:+.2f}R"

def run_backtest():
    global MIN_SCORE
    old_min = MIN_SCORE
    MIN_SCORE = BACKTEST_MIN_SCORE

    logger.info("=" * 74)
    logger.info(f"BINANCE BACKTEST | days={BACKTEST_DAYS} | TFs={BACKTEST_TFS} | MIN_SCORE={BACKTEST_MIN_SCORE}")
    logger.info("=" * 74)

    client = BinanceClient()
    all_trades = []

    for coin in WATCHLIST:
        for tf in BACKTEST_TFS:
            if tf not in TIMEFRAMES: continue
            logger.info(f"--- {coin} {tf} --- ЗАГРУЗКА...")
            try:
                frame = load_backtest_frame(client, coin, tf, BACKTEST_DAYS)
            except Exception as exc:
                logger.error(f"  SKIP: {exc}")
                continue

            last_signal = {}
            signals = 0
            for i in range(REQUIRED_BARS, len(frame) - 1):
                start_idx = max(0, i - 200)
                window = frame.iloc[start_idx:i + 1]
                try:
                    candidates, _ = analyze_orderflow(window)
                except Exception:
                    continue
                if not candidates or len(candidates) > 1: continue

                sig = candidates[0]
                candle_ts = int(frame.index[i])
                if (sig["direction"],) in last_signal and (candle_ts - last_signal[(sig["direction"],)] < COOLDOWN_BARS * TIMEFRAMES[tf][0]): continue

                outcome, pnl_r, bars = simulate_trade(frame, i, sig["direction"], BACKTEST_TP_ATR, BACKTEST_SL_ATR, BACKTEST_MAX_BARS)
                if outcome == "invalid": continue

                last_signal[(sig["direction"],)] = candle_ts
                signals += 1
                all_trades.append({"coin": coin, "tf": tf, "direction": sig["direction"], "score": sig["score"], "setup": sig["setup"], "outcome": outcome, "pnl_r": round(float(pnl_r), 4), "bars_held": int(bars)})
            logger.info(f"  Сигналов: {signals}")

    MIN_SCORE = old_min
    if not all_trades:
        logger.info("=== НЕТ СДЕЛОК ===")
        return

    logger.info("=" * 74)
    logger.info("BACKTEST SUMMARY")
    logger.info(_fmt("ALL", _summary(all_trades)))
    for coin in WATCHLIST:
        logger.info(_fmt(coin, _summary([t for t in all_trades if t["coin"] == coin])))
    logger.info("=" * 74)


# ============================================================
# ХРАНЕНИЕ, TELEGRAM И ОСНОВНОЙ ЦИКЛ
# ============================================================

class StateStore:
    def __init__(self):
        self.db = sqlite3.connect(STATE_DB)
        self.last_cleanup = 0
        self.db.execute("CREATE TABLE IF NOT EXISTS of_signals (id TEXT PRIMARY KEY, coin TEXT, timeframe TEXT, direction TEXT, candle INTEGER, mode TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS of_audit (created INTEGER, coin TEXT, timeframe TEXT, candle INTEGER, result TEXT)")
        self.db.commit()

    def record(self, coin, tf, candle, result):
        serialized = json.dumps(result, ensure_ascii=False, default=str)
        now = int(time.time())
        self.db.execute("INSERT INTO of_audit VALUES (?, ?, ?, ?, ?)", (now, coin, tf, int(candle), serialized))
        if now - self.last_cleanup > 3600:
            self.db.execute("DELETE FROM of_audit WHERE created < ?", (now - 30 * 86400,))
            self.last_cleanup = now
        self.db.commit()
        logger.info("%s %s candle=%s | %s", coin, tf, candle, serialized)

    def signal_id(self, coin, tf, direction, candle):
        mode = "dry" if DRY_RUN else "live"
        return f"{coin}:{tf}:{direction}:{candle}:{mode}"

    def already_sent(self, coin, tf, direction, candle):
        return self.db.execute("SELECT 1 FROM of_signals WHERE id=?", (self.signal_id(coin, tf, direction, candle),)).fetchone() is not None

    def cooldown_active(self, coin, tf, direction, candle):
        mode = "dry" if DRY_RUN else "live"
        row = self.db.execute("SELECT MAX(candle) FROM of_signals WHERE coin=? AND timeframe=? AND direction=? AND mode=?", (coin, tf, direction, mode)).fetchone()
        return row[0] is not None and (candle - row[0]) < COOLDOWN_BARS * TIMEFRAMES[tf][0]

    def mark_sent(self, coin, tf, direction, candle):
        mode = "dry" if DRY_RUN else "live"
        self.db.execute("INSERT OR IGNORE INTO of_signals VALUES (?, ?, ?, ?, ?, ?)", (self.signal_id(coin, tf, direction, candle), coin, tf, direction, int(candle), mode))
        self.db.commit()

def send_telegram(text):
    if DRY_RUN:
        logger.info("DRY_RUN message: %s", text)
        return True
    try:
        res = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=20)
        return res.status_code == 200
    except Exception:
        return False

def format_signal(coin, tf, candle, signal, btc_context, higher_context):
    closed_at = datetime.fromtimestamp(candle + TIMEFRAMES[tf][0], timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    emoji = "🟢" if signal["direction"] == "LONG" else "🔴"
    return NL.join([
        f"{emoji} <b>ORDER FLOW | {tf.upper()} {signal['direction']}</b>", "",
        f"Монета: <b>#{coin}USDT (Binance)</b>", f"Цена закрытия: {signal['price']:,.4f}",
        f"Свеча закрылась: {closed_at}", f"Сетап: {signal['setup']}",
        f"Баллы условий: {signal['score']}/10", "",
        f"📊 Spot delta: {signal['spot_delta_pct']:+.1f}%", f"📊 Perp delta: {signal['perp_delta_pct']:+.1f}%",
        f"🏦 OI: {signal['oi_change']:+.2f}%", f"⚖️ L/S Ratio: {signal['ls_ratio']:.2f}" if signal['ls_ratio'] else "⚖️ L/S: нет", "",
        f"• Spot подтверждает", f"• {signal['reasons'][1]}"
    ])

def get_boundaries():
    eff = int(time.time()) - CLOSE_DELAY
    return {tf: (eff // s[0]) * s[0] for tf, s in TIMEFRAMES.items()}

def check_contexts(coin, tf, direction, cache, boundaries):
    opposite = "bearish" if direction == "LONG" else "bullish"
    btc_tf = HIGHER_TIMEFRAME.get(tf, tf)
    btc_context = "neutral"
    if coin != "BTC":
        rec = cache.get(("BTC", btc_tf))
        if not rec or rec["boundary"] != boundaries[btc_tf]: return "BTC-контекст отсутствует", None, None, True
        btc_context = rec["context"]
    higher_tf = HIGHER_TIMEFRAME.get(tf)
    higher_context = "not_used"
    if higher_tf:
        rec = cache.get((coin, higher_tf))
        if not rec or rec["boundary"] != boundaries[higher_tf]: return "Старший ТФ отсутствует", btc_context, None, True
        higher_context = rec["context"]
        if higher_context == opposite and btc_context == opposite: return "BTC и старший ТФ против", btc_context, higher_context, False
    return None, btc_context, higher_context, False


def main():
    if RUN_BACKTEST:
        threading.Thread(target=start_health_server, daemon=True).start()
        threading.Thread(target=self_ping_worker, daemon=True).start()
        run_backtest()
        return

    if not DRY_RUN and (not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID):
        raise RuntimeError("Задайте токены Telegram")

    threading.Thread(target=start_health_server, daemon=True).start()
    threading.Thread(target=self_ping_worker, daemon=True).start()

    client = BinanceClient()
    state = StateStore()
    
    cache, data_retry_after, telegram_retry_after, processed = {}, {}, {}, set()

    send_telegram("🚀 <b>Binance Order Flow запущен</b>\n" + ", ".join(WATCHLIST) + "\nРежим: " + ("DRY RUN" if DRY_RUN else "LIVE"))
    HEALTH["status"] = "running"

    while True:
        boundaries = get_boundaries()
        cycle_has_errors = False

        for tf in LOAD_ORDER:
            for coin in WATCHLIST:
                key = (coin, tf)
                if cache.get(key) and cache[key]["boundary"] == boundaries[tf]: continue
                if time.time() < data_retry_after.get(key, 0): continue

                try:
                    frame = load_market_data(client, coin, tf, boundaries[tf])
                    cache[key] = {"boundary": boundaries[tf], "frame": frame, "context": get_context(frame)}
                    data_retry_after.pop(key, None)
                    HEALTH["last_data_success"] = int(time.time())
                    logger.info("DATA OK | %s %s | bars=%s", coin, tf, len(frame))
                except Exception as exc:
                    cycle_has_errors = True
                    data_retry_after[key] = time.time() + DATA_RETRY_SECONDS
                    HEALTH["last_error"] = str(exc)
                    logger.warning("DATA ERR | %s %s | %s", coin, tf, exc)

        for tf in LOAD_ORDER:
            seconds = TIMEFRAMES[tf][0]
            for coin in WATCHLIST:
                rec = cache.get((coin, tf))
                if not rec or rec["boundary"] != boundaries[tf]: continue

                frame = rec["frame"]
                candle = int(frame.index[-1])
                work_key = (coin, tf, candle)
                if work_key in processed: continue

                candidates, explanation = analyze_orderflow(frame)
                if not candidates:
                    state.record(coin, tf, candle, {"skip": explanation})
                    processed.add(work_key)
                    continue
                
                if len(candidates) > 1:
                    state.record(coin, tf, candle, {"skip": "Противоречивые"})
                    processed.add(work_key)
                    continue

                signal = candidates[0]
                direction = signal["direction"]
                block_reason, btc_ctx, high_ctx, waiting = check_contexts(coin, tf, direction, cache, boundaries)

                if block_reason:
                    state.record(coin, tf, candle, {"skip": block_reason})
                    if not waiting: processed.add(work_key)
                    continue

                if state.already_sent(coin, tf, direction, candle) or state.cooldown_active(coin, tf, direction, candle):
                    processed.add(work_key)
                    continue

                message = format_signal(coin, tf, candle, signal, btc_ctx, high_ctx)
                if send_telegram(message):
                    state.mark_sent(coin, tf, direction, candle)
                    state.record(coin, tf, candle, {"signal": signal})
                    processed.add(work_key)

        cutoff = int(time.time()) - 3 * 86400
        processed = {k for k in processed if k[2] >= cutoff}
        HEALTH["last_cycle"] = int(time.time())
        if not cycle_has_errors: HEALTH["last_error"] = None
        time.sleep(30)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot stopped")
