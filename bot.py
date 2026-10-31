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
    for value in os.getenv("WATCHLIST", "BTC,ETH,SOL").split(",")
    if value.strip()
]

WATCHLIST = list(dict.fromkeys(["BTC"] + coins))

TIMEFRAMES = {
    "15m": (900, "15min"),
    "1h": (3600, "1hour"),
    "4h": (14400, "4hour"),
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
MIN_LIQ_USD = float(os.getenv("MIN_LIQUIDATION_USD", "10000"))
LIQ_MULTIPLIER = float(os.getenv("LIQUIDATION_SPIKE_MULTIPLIER", "3.0"))

DATA_RETRY_SECONDS = 120
TELEGRAM_RETRY_SECONDS = 60

STATE_DB = os.getenv("STATE_DB", "bot_state.sqlite3")
PREFERRED_EXCHANGE = os.getenv("PREFERRED_EXCHANGE", "").strip()
MARKET_OVERRIDES = json.loads(os.getenv("MARKET_OVERRIDES_JSON", "{}"))

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
}


# ============================================================
# HEALTH-СЕРВЕР
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


# ============================================================
# COINALYZE
# ============================================================

class DataError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class CoinalyzeClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "api_key": COINALYZE_API_KEY,
            "Accept": "application/json",
        })
        self.calls = deque()
        self.last_call = 0.0

    def wait_for_slot(self):
        # Один поток, один символ на запрос.
        # Не более 30 запросов за окно 61 секунда.
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

        if "," in str(params.get("symbols", "")):
            raise ValueError("Допускается один символ на запрос")

        for attempt in range(4):
            self.wait_for_slot()

            try:
                response = self.session.get(
                    f"{API_BASE}/{endpoint}",
                    params=params,
                    timeout=25,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "Network error | %s | %s",
                    endpoint,
                    type(exc).__name__,
                )
                time.sleep(2 ** attempt)
                continue

            if response.status_code == 401:
                raise AuthenticationError("Проверьте COINALYZE_API_KEY")

            if response.status_code == 429:
                try:
                    delay = float(response.headers.get("Retry-After", "60"))
                except ValueError:
                    delay = 60

                logger.warning("Coinalyze 429. Waiting %.1f sec", delay)
                time.sleep(max(1, delay))
                continue

            if response.status_code >= 500:
                time.sleep(2 ** attempt)
                continue

            if response.status_code != 200:
                raise DataError(
                    f"{endpoint}: HTTP {response.status_code}"
                )

            try:
                result = response.json()
            except ValueError:
                time.sleep(2 ** attempt)
                continue

            if not isinstance(result, list):
                raise DataError(f"{endpoint}: ожидался JSON-массив")

            return result

        raise DataError(f"{endpoint}: исчерпаны повторные попытки")

    def history(self, endpoint, symbol, tf, boundary, columns, usd=False):
        seconds, interval = TIMEFRAMES[tf]

        params = {
            "symbols": symbol,
            "interval": interval,
            "from": boundary - LOOKBACK_BARS * seconds,
            "to": boundary - 1,
        }

        if usd:
            params["convert_to_usd"] = "true"

        result = self.get(endpoint, params)

        entry = next(
            (item for item in result if item.get("symbol") == symbol),
            None,
        )

        if entry is None or not entry.get("history"):
            raise DataError(f"{endpoint}: нет истории {symbol}")

        frame = pd.DataFrame(entry["history"])
        required = ["t"] + columns

        if not set(required).issubset(frame.columns):
            raise DataError(f"{endpoint}: отсутствуют поля {required}")

        frame = frame[required].copy()

        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if (
            frame.isna().any().any()
            or not np.isfinite(frame.to_numpy(dtype=float)).all()
        ):
            raise DataError(f"{endpoint}: некорректные числа")

        frame["t"] = frame["t"].astype("int64")
        frame = frame.drop_duplicates("t").set_index("t").sort_index()

        return frame.loc[frame.index + seconds <= boundary]


def build_market_map(client):
    spots = client.get("spot-markets")
    futures = client.get("future-markets")
    mapping = {}

    for coin in WATCHLIST:
        override = MARKET_OVERRIDES.get(coin, {})

        perps = [
            market for market in futures
            if market.get("base_asset") == coin
            and market.get("quote_asset") == "USDT"
            and market.get("is_perpetual") is True
            and market.get("has_ohlcv_data") is True
            and market.get("has_buy_sell_data") is True
        ]

        spot_candidates = [
            market for market in spots
            if market.get("base_asset") == coin
            and market.get("quote_asset") == "USDT"
            and market.get("has_buy_sell_data") is True
        ]

        if override.get("perp"):
            perps = [
                market for market in perps
                if market["symbol"] == override["perp"]
            ]

        if override.get("spot"):
            spot_candidates = [
                market for market in spot_candidates
                if market["symbol"] == override["spot"]
            ]

        if not perps or not spot_candidates:
            raise DataError(
                f"{coin}: не найдены Spot/Perp с buy/sell данными"
            )

        perps.sort(key=lambda market: (
            bool(PREFERRED_EXCHANGE)
            and market.get("exchange") != PREFERRED_EXCHANGE,
            not market.get("has_long_short_ratio_data", False),
            market["symbol"],
        ))

        perp = perps[0]

        spot_candidates.sort(key=lambda market: (
            market.get("exchange") != perp.get("exchange"),
            market["symbol"],
        ))

        mapping[coin] = {
            "spot": spot_candidates[0],
            "perp": perp,
        }

        logger.info(
            "%s | spot=%s | perp=%s | L/S=%s",
            coin,
            mapping[coin]["spot"]["symbol"],
            perp["symbol"],
            perp.get("has_long_short_ratio_data", False),
        )

    return mapping


# ============================================================
# ИНДИКАТОРЫ И СИНХРОНИЗАЦИЯ
# ============================================================

def add_indicators(frame):
    frame = frame.copy()

    frame["spot_delta"] = (
        2 * frame["spot_buy_volume"] - frame["spot_volume"]
    )
    frame["perp_delta"] = 2 * frame["bv"] - frame["v"]

    frame["spot_cvd"] = frame["spot_delta"].cumsum()
    frame["perp_cvd"] = frame["perp_delta"].cumsum()

    frame["spot_share"] = (
        frame["spot_delta"]
        / frame["spot_volume"].replace(0, np.nan)
    )

    frame["perp_share"] = (
        frame["perp_delta"]
        / frame["v"].replace(0, np.nan)
    )

    previous_close = frame["c"].shift(1)

    tr = pd.concat([
        frame["h"] - frame["l"],
        (frame["h"] - previous_close).abs(),
        (frame["l"] - previous_close).abs(),
    ], axis=1).max(axis=1)

    frame["atr"] = tr.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    frame["ema20"] = frame["c"].ewm(
        span=20,
        adjust=False,
    ).mean()

    return frame


def load_market_data(client, market, tf, boundary):
    perp_symbol = market["perp"]["symbol"]
    spot_symbol = market["spot"]["symbol"]

    perp = client.history(
        "ohlcv-history",
        perp_symbol,
        tf,
        boundary,
        ["o", "h", "l", "c", "v", "bv"],
    )

    spot = client.history(
        "ohlcv-history",
        spot_symbol,
        tf,
        boundary,
        ["v", "bv"],
    )

    oi = client.history(
        "open-interest-history",
        perp_symbol,
        tf,
        boundary,
        ["c"],
    )

    funding = client.history(
        "funding-rate-history",
        perp_symbol,
        tf,
        boundary,
        ["c"],
    )

    liquidations = client.history(
        "liquidation-history",
        perp_symbol,
        tf,
        boundary,
        ["l", "s"],
        usd=True,
    )

    frame = perp.join(
        spot.rename(columns={
            "v": "spot_volume",
            "bv": "spot_buy_volume",
        }),
        how="inner",
    )

    frame = frame.join(
        oi.rename(columns={"c": "oi"}),
        how="inner",
    )

    frame = frame.join(
        funding.rename(columns={"c": "funding"}),
        how="inner",
    )

    frame = frame.join(
        liquidations.rename(columns={
            "l": "long_liq",
            "s": "short_liq",
        }),
        how="inner",
    )

    seconds = TIMEFRAMES[tf][0]

    if len(frame) < REQUIRED_BARS:
        raise DataError(
            f"Недостаточно синхронизированных свечей: {len(frame)}"
        )

    if int(frame.index[-1]) != boundary - seconds:
        raise DataError("Последняя закрытая свеча ещё недоступна")

    if not np.all(
        np.diff(frame.index[-REQUIRED_BARS:].to_numpy()) == seconds
    ):
        raise DataError("Пропуски временных интервалов")

    for total, buy in [
        ("v", "bv"),
        ("spot_volume", "spot_buy_volume"),
    ]:
        invalid = (
            (frame[total] < 0)
            | (frame[buy] < 0)
            | (frame[buy] > frame[total] * 1.000001)
        )

        if invalid.any():
            raise DataError("Некорректные объёмы")

    if (frame["oi"] <= 0).any():
        raise DataError("Некорректный Open Interest")

    if (frame[["long_liq", "short_liq"]] < 0).any().any():
        raise DataError("Некорректные ликвидации")

    invalid_ohlc = (
        (frame["h"] < frame[["o", "c", "l"]].max(axis=1))
        | (frame["l"] > frame[["o", "c", "h"]].min(axis=1))
        | (frame["c"] <= 0)
    )

    if invalid_ohlc.any():
        raise DataError("Некорректные OHLC")

    frame["ls_ratio"] = np.nan

    if market["perp"].get("has_long_short_ratio_data", False):
        try:
            ratio = client.history(
                "long-short-ratio-history",
                perp_symbol,
                tf,
                boundary,
                ["r"],
            )

            frame["ls_ratio"] = (
                ratio["r"].where(ratio["r"] > 0).reindex(frame.index)
            )

        except DataError as exc:
            logger.warning(
                "L/S unavailable | %s | %s",
                perp_symbol,
                str(exc),
            )

    return add_indicators(frame)


def get_context(frame):
    current = frame.iloc[-1]
    spot_flow = frame["spot_delta"].iloc[-3:].sum()
    perp_flow = frame["perp_delta"].iloc[-3:].sum()

    if (
        current["c"] > current["ema20"]
        and spot_flow > 0
        and perp_flow > 0
    ):
        return "bullish"

    if (
        current["c"] < current["ema20"]
        and spot_flow < 0
        and perp_flow < 0
    ):
        return "bearish"

    return "neutral"


def detect_divergence(frame, bullish):
    prices = (
        frame["l"].to_numpy()
        if bullish
        else frame["h"].to_numpy()
    )

    cvd = frame["spot_cvd"].to_numpy()
    pivots = []

    for index in range(max(2, len(frame) - 45), len(frame) - 2):
        neighbours = np.concatenate([
            prices[index - 2:index],
            prices[index + 1:index + 3],
        ])

        is_pivot = (
            prices[index] < neighbours.min()
            if bullish
            else prices[index] > neighbours.max()
        )

        if is_pivot:
            pivots.append(index)

    if len(pivots) < 2:
        return False

    first, second = pivots[-2:]

    if second - first < 3 or len(frame) - 1 - second > 7:
        return False

    if bullish:
        return bool(
            prices[second] < prices[first]
            and cvd[second] > cvd[first]
        )

    return bool(
        prices[second] > prices[first]
        and cvd[second] < cvd[first]
    )


# ============================================================
# АНАЛИЗ СИГНАЛОВ
# ============================================================

def analyze_orderflow(frame):
    current = frame.iloc[-1]
    previous = frame.iloc[-2]
    atr = float(frame["atr"].iloc[-2])

    if not np.isfinite(atr) or atr <= 0:
        return [], "Некорректный ATR"

    candle_range = current["h"] - current["l"]

    if candle_range < atr * 0.35:
        return [], "Слишком маленькая свеча по ATR"

    if candle_range > atr * 3:
        return [], "Слишком растянутая свеча по ATR"

    average_volume = frame["v"].iloc[-21:-1].mean()

    if (
        average_volume <= 0
        or current["v"] <= 0
        or current["spot_volume"] <= 0
    ):
        return [], "Недостаточный объём"

    high_volume = current["v"] > average_volume * VOLUME_MULTIPLIER

    prior_low = frame["l"].iloc[-21:-1].min()
    prior_high = frame["h"].iloc[-21:-1].max()

    swept_low = current["l"] < prior_low and current["c"] > prior_low
    swept_high = current["h"] > prior_high and current["c"] < prior_high

    bullish_fvg = current["l"] > frame["h"].iloc[-3]
    bearish_fvg = current["h"] < frame["l"].iloc[-3]

    oi_change = (
        current["oi"] / frame["oi"].iloc[-4] - 1
    ) * 100

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

    # Косвенные признаки возможного поглощения.
    bullish_absorption = (
        current["perp_share"] < -0.10
        and current["c"] >= previous["c"]
    )

    bearish_absorption = (
        current["perp_share"] > 0.10
        and current["c"] <= previous["c"]
    )

    average_long_liq = frame["long_liq"].iloc[-31:-1].mean()
    average_short_liq = frame["short_liq"].iloc[-31:-1].mean()

    long_liq_spike = (
        current["long_liq"] >= MIN_LIQ_USD
        and current["long_liq"]
        > LIQ_MULTIPLIER * max(average_long_liq, 1.0)
    )

    short_liq_spike = (
        current["short_liq"] >= MIN_LIQ_USD
        and current["short_liq"]
        > LIQ_MULTIPLIER * max(average_short_liq, 1.0)
    )

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

        funding = float(current["funding"])

        funding_overheated = (
            funding > FUNDING_LIMIT
            if is_long
            else funding < -FUNDING_LIMIT
        )

        if funding_overheated:
            rejected.append(f"{direction}: funding вне лимита")
            continue

        if is_long:
            recovery = (
                current["c"] > current["o"]
                and current["c"] > previous["c"]
            )
            breakout = current["c"] > prior_high + 0.1 * atr
            reversal_liq = long_liq_spike
            impulse_liq = short_liq_spike

        else:
            recovery = (
                current["c"] < current["o"]
                and current["c"] < previous["c"]
            )
            breakout = current["c"] < prior_low - 0.1 * atr
            reversal_liq = short_liq_spike
            impulse_liq = long_liq_spike

        reversal = bool(
            recovery
            and (sweep or divergence)
            and (absorption or oi_flush or reversal_liq)
        )

        impulse = bool(
            breakout
            and perp_confirms
            and oi_buildup
            and high_volume
        )

        if not reversal and not impulse:
            rejected.append(f"{direction}: нет ценового сценария")
            continue

        setup = (
            "Разворот после свипа/дивергенции"
            if reversal
            else "Подтверждённый пробой"
        )

        score = 4

        reasons = [
            "Spot: дельта и поток за 3 свечи подтверждают",
            (
                "Цена подтвердила разворот у уровня"
                if reversal
                else "Пробой подтверждён CVD Perp, OI и объёмом"
            ),
        ]

        if perp_confirms or (reversal and absorption):
            score += 1
            reasons.append(
                "Perp-поток подтверждает"
                if perp_confirms
                else "Возможное поглощение агрессивного потока"
            )

        if (impulse and oi_buildup) or (reversal and oi_flush):
            score += 1
            reasons.append(
                "OI: рост открытых позиций"
                if oi_buildup
                else "OI: сокращение открытых позиций"
            )

        if divergence:
            score += 1
            reasons.append("Дивергенция цены и Spot CVD")

        if high_volume:
            score += 1
            reasons.append("Объём выше среднего")

        if reversal_liq if reversal else impulse_liq:
            score += 1
            reasons.append("Ликвидации аномальны относительно истории")

        ratio = (
            float(current["ls_ratio"])
            if pd.notna(current["ls_ratio"])
            else None
        )

        ratio_confirms = (
            ratio is not None
            and (
                (is_long and ratio < 0.8)
                or (not is_long and ratio > 1.25)
            )
        )

        if ratio_confirms:
            score += 1
            reasons.append("L/S: контрнаправленный перекос")

        if score < MIN_SCORE:
            rejected.append(
                f"{direction}: score={score}, нужно {MIN_SCORE}"
            )
            continue

        candidates.append({
            "direction": direction,
            "setup": setup,
            "score": int(score),
            "reasons": reasons,
            "price": float(current["c"]),
            "atr": atr,
            "oi_change": float(oi_change),
            "funding": funding,
            "spot_delta_pct": float(current["spot_share"] * 100),
            "perp_delta_pct": float(current["perp_share"] * 100),
            "long_liq": float(current["long_liq"]),
            "short_liq": float(current["short_liq"]),
            "ls_ratio": ratio,
            "fvg": bool(bullish_fvg if is_long else bearish_fvg),
        })

    return candidates, "; ".join(rejected)


# ============================================================
# ХРАНЕНИЕ СОСТОЯНИЯ
# ============================================================

class StateStore:
    def __init__(self):
        self.db = sqlite3.connect(STATE_DB)
        self.last_cleanup = 0

        self.db.execute(
            "CREATE TABLE IF NOT EXISTS of_signals ("
            "id TEXT PRIMARY KEY, "
            "coin TEXT NOT NULL, "
            "timeframe TEXT NOT NULL, "
            "direction TEXT NOT NULL, "
            "candle INTEGER NOT NULL, "
            "mode TEXT NOT NULL)"
        )

        self.db.execute(
            "CREATE TABLE IF NOT EXISTS of_audit ("
            "created INTEGER NOT NULL, "
            "coin TEXT NOT NULL, "
            "timeframe TEXT NOT NULL, "
            "candle INTEGER NOT NULL, "
            "result TEXT NOT NULL)"
        )

        self.db.commit()

    def record(self, coin, tf, candle, result):
        serialized = json.dumps(result, ensure_ascii=False)
        now = int(time.time())

        self.db.execute(
            "INSERT INTO of_audit VALUES (?, ?, ?, ?, ?)",
            (now, coin, tf, int(candle), serialized),
        )

        if now - self.last_cleanup > 3600:
            self.db.execute(
                "DELETE FROM of_audit WHERE created < ?",
                (now - 30 * 86400,),
            )
            self.last_cleanup = now

        self.db.commit()

        logger.info(
            "%s %s candle=%s | %s",
            coin, tf, candle, serialized,
        )

    def signal_id(self, coin, tf, direction, candle):
        mode = "dry" if DRY_RUN else "live"
        return f"{coin}:{tf}:{direction}:{candle}:{mode}"

    def already_sent(self, coin, tf, direction, candle):
        row = self.db.execute(
            "SELECT 1 FROM of_signals WHERE id=?",
            (self.signal_id(coin, tf, direction, candle),),
        ).fetchone()

        return row is not None

    def cooldown_active(self, coin, tf, direction, candle):
        mode = "dry" if DRY_RUN else "live"

        row = self.db.execute(
            "SELECT MAX(candle) FROM of_signals "
            "WHERE coin=? AND timeframe=? AND direction=? AND mode=?",
            (coin, tf, direction, mode),
        ).fetchone()

        previous_candle = row[0]

        if previous_candle is None:
            return False

        return (
            candle - previous_candle
            < COOLDOWN_BARS * TIMEFRAMES[tf][0]
        )

    def mark_sent(self, coin, tf, direction, candle):
        mode = "dry" if DRY_RUN else "live"

        self.db.execute(
            "INSERT OR IGNORE INTO of_signals "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                self.signal_id(coin, tf, direction, candle),
                coin,
                tf,
                direction,
                int(candle),
                mode,
            ),
        )

        self.db.commit()


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(text):
    if DRY_RUN:
        logger.info("DRY_RUN message: %s", text)
        return True

    url = (
        "https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    try:
        response = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
            },
            timeout=20,
        )

        result = response.json()

        if response.status_code == 200 and result.get("ok") is True:
            return True

        logger.error(
            "Telegram failed | HTTP=%s | error_code=%s",
            response.status_code,
            result.get("error_code"),
        )

    except (requests.RequestException, ValueError) as exc:
        logger.error(
            "Telegram exception: %s",
            type(exc).__name__,
        )

    return False


def format_signal(
    coin, tf, candle, signal, btc_context, higher_context, market
):
    closed_at = datetime.fromtimestamp(
        candle + TIMEFRAMES[tf][0],
        timezone.utc,
    ).strftime("%Y-%m-%d %H:%M UTC")

    ratio_text = (
        "недоступен"
        if signal["ls_ratio"] is None
        else f'{signal["ls_ratio"]:.2f}'
    )

    context_names = {
        "bullish": "бычий",
        "bearish": "медвежий",
        "neutral": "нейтральный",
        "not_used": "не используется",
    }

    emoji = "🟢" if signal["direction"] == "LONG" else "🔴"
    fvg_text = "да" if signal["fvg"] else "нет"

    lines = [
        f"{emoji} <b>ORDER FLOW | {tf.upper()} {signal['direction']}</b>",
        "",
        f"Монета: <b>#{html.escape(coin)}USDT</b>",
        f"Цена закрытия: {signal['price']:,.4f}",
        f"Свеча закрылась: {closed_at}",
        f"Сетап: {html.escape(signal['setup'])}",
        f"Баллы условий: {signal['score']}/10",
        "<i>Баллы — не вероятность успеха.</i>",
        "",
        f"📊 Spot delta / volume: {signal['spot_delta_pct']:+.1f}%",
        f"📊 Perp delta / volume: {signal['perp_delta_pct']:+.1f}%",
        f"🏦 OI за 3 свечи: {signal['oi_change']:+.2f}%",
        f"💰 Funding: {signal['funding']:+.4f}%",
        f"💥 Ликвидации LONG: ${signal['long_liq']:,.0f}",
        f"💥 Ликвидации SHORT: ${signal['short_liq']:,.0f}",
        f"⚖️ Long/Short Ratio: {ratio_text}",
        f"📏 ATR(14): {signal['atr']:,.4f}",
        f"📐 FVG по направлению: {fvg_text}",
        "",
        f"BTC: {context_names.get(btc_context, btc_context)}",
        f"Старший ТФ: {context_names.get(higher_context, higher_context)}",
        "",
    ]

    lines.extend(
        "• " + html.escape(reason)
        for reason in signal["reasons"]
    )

    lines.extend([
        "",
        "Источник: Coinalyze",
        "Spot: " + html.escape(market["spot"]["symbol"]),
        "Perp: " + html.escape(market["perp"]["symbol"]),
        (
            "<i>Один рынок каждого типа, не агрегат всех бирж. "
            "Уведомление о сетапе, не команда на вход.</i>"
        ),
    ])

    return NL.join(lines)


# ============================================================
# КОНТЕКСТ BTC И СТАРШЕГО ТФ
# ============================================================

def get_boundaries():
    effective_time = int(time.time()) - CLOSE_DELAY

    return {
        tf: (effective_time // settings[0]) * settings[0]
        for tf, settings in TIMEFRAMES.items()
    }


def check_contexts(coin, tf, direction, cache, boundaries):
    opposite = "bearish" if direction == "LONG" else "bullish"
    btc_record = cache.get(("BTC", tf))

    if coin != "BTC":
        if (
            btc_record is None
            or btc_record["boundary"] != boundaries[tf]
        ):
            return "BTC-контекст отсутствует", None, None, True

        if btc_record["context"] == opposite:
            return (
                "BTC-контекст против сигнала",
                btc_record["context"],
                None,
                False,
            )

    higher_tf = HIGHER_TIMEFRAME.get(tf)
    higher_context = "not_used"

    if higher_tf:
        higher_record = cache.get((coin, higher_tf))

        if (
            higher_record is None
            or higher_record["boundary"] != boundaries[higher_tf]
        ):
            return "Старший ТФ отсутствует", None, None, True

        higher_context = higher_record["context"]

        if higher_context == opposite:
            return (
                "Старший ТФ против сигнала",
                None,
                higher_context,
                False,
            )

    btc_context = (
        btc_record["context"] if btc_record else "neutral"
    )

    return None, btc_context, higher_context, False


# ============================================================
# ГЛАВНЫЙ ЦИКЛ
# ============================================================

def main():
    if not COINALYZE_API_KEY:
        raise RuntimeError(
            "Задайте COINALYZE_API_KEY в Render Environment"
        )

    if not DRY_RUN and (
        not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID
    ):
        raise RuntimeError(
            "Задайте TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID"
        )

    if not 1 <= MIN_SCORE <= 10:
        raise RuntimeError("MIN_SCORE должен быть от 1 до 10")

    threading.Thread(
        target=start_health_server,
        daemon=True,
    ).start()

    client = CoinalyzeClient()
    state = StateStore()

    while True:
        try:
            market_map = build_market_map(client)
            break

        except AuthenticationError:
            raise

        except DataError as exc:
            HEALTH["last_error"] = str(exc)
            logger.error(
                "Market mapping failed: %s. Retry in 120 sec",
                str(exc),
            )
            time.sleep(120)

    cache = {}
    data_retry_after = {}
    telegram_retry_after = {}
    processed = set()

    mode = (
        "DRY RUN: сообщения только в логах"
        if DRY_RUN
        else "LIVE: отправка в Telegram"
    )

    send_telegram(NL.join([
        "🚀 <b>Coinalyze Order Flow запущен</b>",
        "Монеты: " + ", ".join(WATCHLIST),
        "Таймфреймы: 15m, 1h, 4h",
        "Режим: " + mode,
        "Первичная загрузка займёт несколько минут.",
    ]))

    HEALTH["status"] = "running"

    while True:
        boundaries = get_boundaries()
        cycle_has_errors = False

        for tf in LOAD_ORDER:
            boundary = boundaries[tf]

            for coin in WATCHLIST:
                key = (coin, tf)
                existing = cache.get(key)

                if (
                    existing is not None
                    and existing["boundary"] == boundary
                ):
                    continue

                if time.time() < data_retry_after.get(key, 0):
                    cycle_has_errors = True
                    continue

                try:
                    frame = load_market_data(
                        client,
                        market_map[coin],
                        tf,
                        boundary,
                    )

                    cache[key] = {
                        "boundary": boundary,
                        "frame": frame,
                        "context": get_context(frame),
                    }

                    data_retry_after.pop(key, None)
                    HEALTH["last_data_success"] = int(time.time())

                    logger.info(
                        "DATA OK | %s %s | bars=%s | context=%s",
                        coin,
                        tf,
                        len(frame),
                        cache[key]["context"],
                    )

                except AuthenticationError:
                    raise

                except DataError as exc:
                    cycle_has_errors = True
                    cache.pop(key, None)

                    data_retry_after[key] = (
                        time.time() + DATA_RETRY_SECONDS
                    )

                    HEALTH["last_error"] = str(exc)

                    state.record(
                        coin,
                        tf,
                        boundary,
                        {"data_error": str(exc)},
                    )

        for tf in LOAD_ORDER:
            seconds = TIMEFRAMES[tf][0]

            for coin in WATCHLIST:
                record = cache.get((coin, tf))

                if (
                    record is None
                    or record["boundary"] != boundaries[tf]
                ):
                    continue

                frame = record["frame"]
                candle = int(frame.index[-1])
                work_key = (coin, tf, candle)

                if work_key in processed:
                    continue

                if (
                    time.time() - (candle + seconds)
                    > MAX_SIGNAL_DELAY
                ):
                    state.record(
                        coin,
                        tf,
                        candle,
                        {
                            "skip": (
                                "Старая свеча: используется "
                                "только как контекст"
                            )
                        },
                    )
                    processed.add(work_key)
                    continue

                candidates, explanation = analyze_orderflow(frame)

                if not candidates:
                    state.record(
                        coin,
                        tf,
                        candle,
                        {
                            "skip": (
                                explanation
                                or "Нет подтверждённого сетапа"
                            )
                        },
                    )
                    processed.add(work_key)
                    continue

                if len(candidates) > 1:
                    state.record(
                        coin,
                        tf,
                        candle,
                        {"skip": "Противоречивые направления"},
                    )
                    processed.add(work_key)
                    continue

                signal = candidates[0]
                direction = signal["direction"]

                (
                    block_reason,
                    btc_context,
                    higher_context,
                    waiting_for_data,
                ) = check_contexts(
                    coin,
                    tf,
                    direction,
                    cache,
                    boundaries,
                )

                if block_reason:
                    state.record(
                        coin,
                        tf,
                        candle,
                        {
                            "skip": block_reason,
                            "candidate": signal,
                        },
                    )

                    if not waiting_for_data:
                        processed.add(work_key)

                    continue

                if state.already_sent(
                    coin, tf, direction, candle
                ):
                    processed.add(work_key)
                    continue

                if state.cooldown_active(
                    coin, tf, direction, candle
                ):
                    state.record(
                        coin,
                        tf,
                        candle,
                        {"skip": "Пауза между сигналами"},
                    )
                    processed.add(work_key)
                    continue

                if time.time() < telegram_retry_after.get(work_key, 0):
                    continue

                message = format_signal(
                    coin,
                    tf,
                    candle,
                    signal,
                    btc_context,
                    higher_context,
                    market_map[coin],
                )

                if send_telegram(message):
                    state.mark_sent(
                        coin, tf, direction, candle
                    )

                    state.record(
                        coin,
                        tf,
                        candle,
                        {
                            "signal": signal,
                            "mode": "dry_run" if DRY_RUN else "telegram",
                        },
                    )

                    processed.add(work_key)
                    telegram_retry_after.pop(work_key, None)

                else:
                    telegram_retry_after[work_key] = (
                        time.time() + TELEGRAM_RETRY_SECONDS
                    )

        cutoff = int(time.time()) - 3 * 86400

        processed = {
            key for key in processed
            if key[2] >= cutoff
        }

        telegram_retry_after = {
            key: value
            for key, value in telegram_retry_after.items()
            if key[2] >= cutoff
        }

        HEALTH["last_cycle"] = int(time.time())

        if not cycle_has_errors:
            HEALTH["last_error"] = None

        time.sleep(30)


if __name__ == "__main__":
    try:
        main()

    except AuthenticationError as exc:
        HEALTH["status"] = "authentication_error"
        HEALTH["last_error"] = str(exc)
        logger.critical("%s", str(exc))
        raise SystemExit(1)

    except KeyboardInterrupt:
        logger.info("Bot stopped")

    except Exception as exc:
        HEALTH["status"] = "fatal_error"
        HEALTH["last_error"] = type(exc).__name__

        logger.critical(
            "Fatal error: %s",
            type(exc).__name__,
        )

        raise SystemExit(1)
