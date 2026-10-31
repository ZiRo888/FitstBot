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
# 1. НАСТРОЙКИ
# ============================================================

COINALYZE_API_KEY = os.getenv("COINALYZE_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

API_BASE = "https://api.coinalyze.net/v1"

# Указывать базовые активы: BTC,ETH,SOL, а не BTCUSDT.
configured_coins = [
    value.strip().upper()
    for value in os.getenv("WATCHLIST", "BTC,ETH,SOL").split(",")
    if value.strip()
]

# BTC нужен для контекста альтов.
WATCHLIST = list(dict.fromkeys(["BTC"] + configured_coins))

TIMEFRAMES = {
    "15m": {"seconds": 900, "api": "15min"},
    "1h": {"seconds": 3600, "api": "1hour"},
    "4h": {"seconds": 14400, "api": "4hour"},
}

LOAD_ORDER = ["4h", "1h", "15m"]

HIGHER_TF = {
    "15m": "1h",
    "1h": "4h",
}

LOOKBACK_BARS = 100
REQUIRED_BARS = 65

CLOSE_DELAY_SECONDS = int(
    os.getenv("CLOSE_DELAY_SECONDS", "45")
)

# Не отправлять давно закрывшиеся сетапы при перезапуске.
MAX_SIGNAL_DELAY_SECONDS = int(
    os.getenv("MAX_SIGNAL_DELAY_SECONDS", "600")
)

DATA_RETRY_SECONDS = 120
TELEGRAM_RETRY_SECONDS = 60

MIN_SCORE = int(os.getenv("MIN_SCORE", "7"))
COOLDOWN_BARS = int(os.getenv("COOLDOWN_BARS", "2"))

# Минимальная доля дельты в объёме: 3%.
MIN_DELTA_SHARE = float(
    os.getenv("MIN_DELTA_SHARE", "0.03")
)

OI_CHANGE_THRESHOLD = float(
    os.getenv("OI_CHANGE_PERCENT", "0.30")
)

VOLUME_MULTIPLIER = float(
    os.getenv("VOLUME_MULTIPLIER", "1.50")
)

# Funding в API уже выражен в процентах:
# 0.05 означает 0.05%, а не 5%.
FUNDING_LIMIT_PERCENT = float(
    os.getenv("FUNDING_LIMIT_PERCENT", "0.05")
)

MIN_LIQUIDATION_USD = float(
    os.getenv("MIN_LIQUIDATION_USD", "10000")
)

LIQUIDATION_SPIKE_MULTIPLIER = float(
    os.getenv("LIQUIDATION_SPIKE_MULTIPLIER", "3.0")
)

ATR_MIN_MULTIPLIER = 0.35
ATR_MAX_MULTIPLIER = 3.0

# Необязательное закрепление конкретных рынков.
# Значение переменной — JSON с реальными символами Coinalyze:
# {"BTC": {"spot": "...", "perp": "..."}}
MARKET_OVERRIDES = json.loads(
    os.getenv("MARKET_OVERRIDES_JSON", "{}")
)

# Если задан код биржи из каталога Coinalyze,
# рынки этой биржи получают приоритет.
PREFERRED_EXCHANGE = os.getenv(
    "PREFERRED_EXCHANGE", ""
).strip()

STATE_DB = os.getenv("STATE_DB", "bot_state.sqlite3")

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
# 2. ИСКЛЮЧЕНИЯ
# ============================================================

class DataError(Exception):
    pass


class AuthenticationError(Exception):
    pass


# ============================================================
# 3. HEALTH-СЕРВЕР
# ============================================================

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps(
            HEALTH, ensure_ascii=False
        ).encode("utf-8")

        self.send_response(200)
        self.send_header(
            "Content-Type", "application/json; charset=utf-8"
        )
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def start_health_server():
    port = int(os.getenv("PORT", "10000"))

    server = HTTPServer(
        ("0.0.0.0", port),
        HealthHandler,
    )

    logger.info("Health server: port %s", port)
    server.serve_forever()


# ============================================================
# 4. COINALYZE: ОБЩИЙ ОГРАНИЧИТЕЛЬ И ЗАПРОСЫ
# ============================================================

class CoinalyzeClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "api_key": COINALYZE_API_KEY,
            "Accept": "application/json",
        })

        self.request_times = deque()
        self.last_request_time = 0.0

    def wait_for_slot(self):
        """
        Клиент используется последовательно, из одного потока.
        Максимум 30 запросов за окно 61 секунда.
        В одном запросе истории — один символ.
        """
        while True:
            now = time.monotonic()

            while (
                self.request_times
                and now - self.request_times[0] >= 61
            ):
                self.request_times.popleft()

            wait = max(
                0.0,
                self.last_request_time + 2.1 - now,
            )

            if len(self.request_times) >= 30:
                wait = max(
                    wait,
                    self.request_times[0] + 61 - now,
                )

            if wait <= 0:
                self.request_times.append(now)
                self.last_request_time = now
                return

            time.sleep(wait + 0.05)

    def get(self, endpoint, params=None):
        params = dict(params or {})

        # Пакеты символов требуют отдельного учёта стоимости.
        if "," in str(params.get("symbols", "")):
            raise ValueError(
                "Этот клиент принимает один символ на запрос"
            )

        for attempt in range(4):
            self.wait_for_slot()

            try:
                response = self.session.get(
                    f"{API_BASE}/{endpoint}",
                    params=params,
                    timeout=25,
                )
            except requests.RequestException as exc:
                # Не печатаем URL или текст исключения с секретами.
                logger.warning(
                    "Network error: endpoint=%s type=%s",
                    endpoint,
                    type(exc).__name__,
                )
                time.sleep(2 ** attempt)
                continue

            if response.status_code == 401:
                raise AuthenticationError(
                    "Coinalyze: неверный или отсутствующий API-ключ"
                )

            if response.status_code == 429:
                try:
                    wait = float(
                        response.headers.get(
                            "Retry-After", "60"
                        )
                    )
                except ValueError:
                    wait = 60

                logger.warning(
                    "Coinalyze rate limit. Waiting %.1f sec",
                    wait,
                )
                time.sleep(max(wait, 1))
                continue

            if response.status_code >= 500:
                time.sleep(2 ** attempt)
                continue

            if response.status_code != 200:
                raise DataError(
                    f"{endpoint}: HTTP {response.status_code}"
                )

            try:
                data = response.json()
            except ValueError:
                time.sleep(2 ** attempt)
                continue

            if not isinstance(data, list):
                raise DataError(
                    f"{endpoint}: ответ должен быть массивом"
                )

            return data

        raise DataError(
            f"{endpoint}: исчерпаны повторные попытки"
        )

    def history(
        self,
        endpoint,
        symbol,
        timeframe,
        boundary,
        columns,
        convert_to_usd=False,
    ):
        settings = TIMEFRAMES[timeframe]
        seconds = settings["seconds"]

        params = {
            "symbols": symbol,
            "interval": settings["api"],
            "from": boundary - LOOKBACK_BARS * seconds,
            "to": boundary - 1,
        }

        if convert_to_usd:
            params["convert_to_usd"] = "true"

        result = self.get(endpoint, params)

        entry = next(
            (
                item for item in result
                if item.get("symbol") == symbol
            ),
            None,
        )

        if entry is None or not entry.get("history"):
            raise DataError(
                f"{endpoint}: нет истории {symbol}"
            )

        frame = pd.DataFrame(entry["history"])

        required = ["t"] + columns

        if not set(required).issubset(frame.columns):
            raise DataError(
                f"{endpoint}: отсутствуют поля {required}"
            )

        frame = frame[required].copy()

        for column in required:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        if (
            frame.isna().any().any()
            or not np.isfinite(
                frame.to_numpy(dtype=float)
            ).all()
        ):
            raise DataError(
                f"{endpoint}: некорректные числовые данные"
            )

        frame["t"] = frame["t"].astype("int64")

        frame = (
            frame.drop_duplicates("t")
            .set_index("t")
            .sort_index()
        )

        # Только полностью закрытые интервалы.
        return frame.loc[
            frame.index + seconds <= boundary
        ]


# ============================================================
# 5. ВЫБОР РЫНКОВ
# ============================================================

def build_market_map(client):
    spots = client.get("spot-markets")
    futures = client.get("future-markets")

    mapping = {}

    for coin in WATCHLIST:
        override = MARKET_OVERRIDES.get(coin, {})

        perp_candidates = [
            market
            for market in futures
            if market.get("base_asset") == coin
            and market.get("quote_asset") == "USDT"
            and market.get("is_perpetual") is True
            and market.get("margined") == "STABLE"
            and market.get("has_ohlcv_data") is True
            and market.get("has_buy_sell_data") is True
        ]

        spot_candidates = [
            market
            for market in spots
            if market.get("base_asset") == coin
            and market.get("quote_asset") == "USDT"
            and market.get("has_buy_sell_data") is True
        ]

        if override.get("perp"):
            perp_candidates = [
                market for market in perp_candidates
                if market["symbol"] == override["perp"]
            ]

        if override.get("spot"):
            spot_candidates = [
                market for market in spot_candidates
                if market["symbol"] == override["spot"]
            ]

        if not perp_candidates or not spot_candidates:
            raise DataError(
                f"{coin}: нет подходящих рынков Spot/Perp. "
                "Проверьте WATCHLIST и MARKET_OVERRIDES_JSON."
            )

        perp_candidates.sort(
            key=lambda market: (
                bool(PREFERRED_EXCHANGE)
                and market.get("exchange")
                != PREFERRED_EXCHANGE,
                not market.get(
                    "has_long_short_ratio_data", False
                ),
                market["symbol"],
            )
        )

        perp = perp_candidates[0]

        spot_candidates.sort(
            key=lambda market: (
                market.get("exchange")
                != perp.get("exchange"),
                market["symbol"],
            )
        )

        spot = spot_candidates[0]

        mapping[coin] = {
            "spot": spot,
            "perp": perp,
        }

        logger.info(
            "%s | spot=%s | perp=%s | L/S=%s",
            coin,
            spot["symbol"],
            perp["symbol"],
            perp.get("has_long_short_ratio_data", False),
        )

    return mapping


# ============================================================
# 6. ЗАГРУЗКА И СИНХРОНИЗАЦИЯ
# ============================================================

def load_market_data(
    client,
    market,
    timeframe,
    boundary,
):
    perp_symbol = market["perp"]["symbol"]
    spot_symbol = market["spot"]["symbol"]

    perp = client.history(
        "ohlcv-history",
        perp_symbol,
        timeframe,
        boundary,
        ["o", "h", "l", "c", "v", "bv"],
    )

    spot = client.history(
        "ohlcv-history",
        spot_symbol,
        timeframe,
        boundary,
        ["v", "bv"],
    )

    oi = client.history(
        "open-interest-history",
        perp_symbol,
        timeframe,
        boundary,
        ["c"],
    )

    funding = client.history(
        "funding-rate-history",
        perp_symbol,
        timeframe,
        boundary,
        ["c"],
    )

    liquidations = client.history(
        "liquidation-history",
        perp_symbol,
        timeframe,
        boundary,
        ["l", "s"],
        convert_to_usd=True,
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

    seconds = TIMEFRAMES[timeframe]["seconds"]

    if len(frame) < REQUIRED_BARS:
        raise DataError(
            f"Недостаточно совпавших свечей: {len(frame)}"
        )

    if int(frame.index[-1]) != boundary - seconds:
        raise DataError(
            "Последняя закрытая свеча ещё не загружена"
        )

    recent_times = frame.index[-REQUIRED_BARS:].to_numpy()

    if not np.all(np.diff(recent_times) == seconds):
        raise DataError(
            "Пропуски временных интервалов"
        )

    for volume, buy in [
        ("v", "bv"),
        ("spot_volume", "spot_buy_volume"),
    ]:
        if (
            (frame[volume] < 0).any()
            or (frame[buy] < 0).any()
            or (
                frame[buy]
                > frame[volume] * 1.000001
            ).any()
        ):
            raise DataError(
                "Некорректный buy/total volume"
            )

    if (
        (frame["oi"] <= 0).any()
        or (
            frame[["long_liq", "short_liq"]] < 0
        ).any().any()
    ):
        raise DataError(
            "Некорректный OI или ликвидации"
        )

    invalid_ohlc = (
        (frame["h"] < frame[["o", "c", "l"]].max(axis=1))
        | (frame["l"] > frame[["o", "c", "h"]].min(axis=1))
        | (frame["c"] <= 0)
    )

    if invalid_ohlc.any():
        raise DataError("Некорректные OHLC")

    # L/S — вспомогательная метрика.
    # Если она недоступна, не выдумываем значение.
    frame["ls_ratio"] = np.nan

    if market["perp"].get(
        "has_long_short_ratio_data", False
    ):
        try:
            ratio = client.history(
                "long-short-ratio-history",
                perp_symbol,
                timeframe,
                boundary,
                ["r"],
            )

            valid_ratio = ratio["r"].where(
                ratio["r"] > 0
            )

            frame["ls_ratio"] = valid_ratio.reindex(
                frame.index
            )

        except DataError as exc:
            logger.warning(
                "L/S unavailable: %s %s",
                perp_symbol,
                str(exc),
            )

    return add_indicators(frame)


# ============================================================
# 7. CVD, ATR И КОНТЕКСТ
# ============================================================

def add_indicators(frame):
    frame = frame.copy()

    frame["spot_delta"] = (
        2 * frame["spot_buy_volume"]
        - frame["spot_volume"]
    )

    frame["perp_delta"] = (
        2 * frame["bv"] - frame["v"]
    )

    frame["spot_cvd"] = frame["spot_delta"].cumsum()
    frame["perp_cvd"] = frame["perp_delta"].cumsum()

    frame["spot_delta_share"] = (
        frame["spot_delta"]
        / frame["spot_volume"].replace(0, np.nan)
    )

    frame["perp_delta_share"] = (
        frame["perp_delta"]
        / frame["v"].replace(0, np.nan)
    )

    previous_close = frame["c"].shift(1)

    true_range = pd.concat(
        [
            frame["h"] - frame["l"],
            (frame["h"] - previous_close).abs(),
            (frame["l"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    frame["atr"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    frame["ema20"] = frame["c"].ewm(
        span=20,
        adjust=False,
    ).mean()

    return frame


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


def detect_cvd_divergence(frame, bullish=True):
    """
    Два подтверждённых ценовых pivot.
    Для подтверждения pivot нужны две свечи справа.
    Поэтому дивергенция определяется с задержкой.
    """
    prices = (
        frame["l"].to_numpy()
        if bullish
        else frame["h"].to_numpy()
    )

    cvd = frame["spot_cvd"].to_numpy()
    pivots = []

    start = max(2, len(frame) - 45)

    for i in range(start, len(frame) - 2):
        neighbours = np.concatenate([
            prices[i - 2:i],
            prices[i + 1:i + 3],
        ])

        is_pivot = (
            prices[i] < neighbours.min()
            if bullish
            else prices[i] > neighbours.max()
        )

        if is_pivot:
            pivots.append(i)

    if len(pivots) < 2:
        return False

    first, second = pivots[-2:]

    if second - first < 3:
        return False

    if len(frame) - 1 - second > 7:
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
# 8. АНАЛИЗ СЦЕНАРИЕВ
# ============================================================

def analyze_orderflow(frame):
    current = frame.iloc[-1]
    previous = frame.iloc[-2]

    # ATR берётся до сигнальной свечи.
    atr = float(frame["atr"].iloc[-2])

    if not np.isfinite(atr) or atr <= 0:
        return [], "Некорректный ATR"

    candle_range = current["h"] - current["l"]

    if candle_range < atr * ATR_MIN_MULTIPLIER:
        return [], "Слишком маленькая свеча относительно ATR"

    if candle_range > atr * ATR_MAX_MULTIPLIER:
        return [], "Свеча чрезмерно растянута относительно ATR"

    average_volume = frame["v"].iloc[-21:-1].mean()

    if (
        average_volume <= 0
        or current["v"] <= 0
        or current["spot_volume"] <= 0
    ):
        return [], "Недостаточный объём"

    high_volume = (
        current["v"]
        > average_volume * VOLUME_MULTIPLIER
    )

    prior_low = frame["l"].iloc[-21:-1].min()
    prior_high = frame["h"].iloc[-21:-1].max()

    # Sweep — не просто касание:
    # выход за уровень и закрытие обратно в диапазоне.
    swept_low = (
        current["l"] < prior_low
        and current["c"] > prior_low
    )

    swept_high = (
        current["h"] > prior_high
        and current["c"] < prior_high
    )

    bullish_fvg = (
        current["l"] > frame["h"].iloc[-3]
    )

    bearish_fvg = (
        current["h"] < frame["l"].iloc[-3]
    )

    oi_change = (
        current["oi"] / frame["oi"].iloc[-4] - 1
    ) * 100

    oi_buildup = (
        oi_change > OI_CHANGE_THRESHOLD
    )

    oi_flush = (
        oi_change < -OI_CHANGE_THRESHOLD
    )

    spot_flow = frame["spot_delta"].iloc[-3:].sum()
    perp_flow = frame["perp_delta"].iloc[-3:].sum()

    spot_up = (
        current["spot_delta_share"] > MIN_DELTA_SHARE
        and spot_flow > 0
    )

    spot_down = (
        current["spot_delta_share"] < -MIN_DELTA_SHARE
        and spot_flow < 0
    )

    perp_up = (
        current["perp_delta_share"] > MIN_DELTA_SHARE
        and perp_flow > 0
    )

    perp_down = (
        current["perp_delta_share"] < -MIN_DELTA_SHARE
        and perp_flow < 0
    )

    bullish_divergence = detect_cvd_divergence(
        frame, bullish=True
    )

    bearish_divergence = detect_cvd_divergence(
        frame, bullish=False
    )

    # Возможное поглощение.
    # Это косвенная оценка, а не наблюдение стакана.
    bullish_absorption = (
        current["perp_delta_share"] < -0.10
        and current["c"] >= previous["c"]
    )

    bearish_absorption = (
        current["perp_delta_share"] > 0.10
        and current["c"] <= previous["c"]
    )

    average_long_liq = (
        frame["long_liq"].iloc[-31:-1].mean()
    )

    average_short_liq = (
        frame["short_liq"].iloc[-31:-1].mean()
    )

    long_liq_spike = (
        current["long_liq"] >= MIN_LIQUIDATION_USD
        and current["long_liq"]
        > LIQUIDATION_SPIKE_MULTIPLIER
        * max(average_long_liq, 1.0)
    )

    short_liq_spike = (
        current["short_liq"] >= MIN_LIQUIDATION_USD
        and current["short_liq"]
        > LIQUIDATION_SPIKE_MULTIPLIER
        * max(average_short_liq, 1.0)
    )

    candidates = []
    rejected = []

    for direction in ["LONG", "SHORT"]:
        is_long = direction == "LONG"

        spot_confirms = spot_up if is_long else spot_down
        perp_confirms = perp_up if is_long else perp_down

        sweep = swept_low if is_long else swept_high

        divergence = (
            bullish_divergence
            if is_long
            else bearish_divergence
        )

        absorption = (
            bullish_absorption
            if is_long
            else bearish_absorption
        )

        if not spot_confirms:
            rejected.append(
                f"{direction}: Spot не подтверждает"
            )
            continue

        funding = float(current["funding"])

        funding_overheated = (
            funding > FUNDING_LIMIT_PERCENT
            if is_long
            else funding < -FUNDING_LIMIT_PERCENT
        )

        if funding_overheated:
            rejected.append(
                f"{direction}: funding вне лимита"
            )
            continue

        recovery = (
            (
                current["c"] > current["o"]
                and current["c"] > previous["c"]
            )
            if is_long
            else (
                current["c"] < current["o"]
                and current["c"] < previous["c"]
            )
        )

        breakout = (
            current["c"] > prior_high + 0.1 * atr
            if is_long
            else current["c"] < prior_low - 0.1 * atr
        )

        reversal_liquidations = (
            long_liq_spike
            if is_long
            else short_liq_spike
        )

        impulse_liquidations = (
            short_liq_spike
            if is_long
            else long_liq_spike
        )

        reversal = bool(
            recovery
            and (sweep or divergence)
            and (
                absorption
                or oi_flush
                or reversal_liquidations
            )
        )

        impulse = bool(
            breakout
            and perp_confirms
            and oi_buildup
            and high_volume
        )

        if not reversal and not impulse:
            rejected.append(
                f"{direction}: нет ценового сценария"
            )
            continue

        setup = (
            "Разворот после свипа/дивергенции"
            if reversal
            else "Подтверждённый пробой"
        )

        # Обязательная база: Spot + валидный сценарий.
        score = 4

        reasons = [
            "Spot: дельта и поток за 3 свечи подтверждают",
            (
                "Цена подтвердила восстановление/отклонение"
                if reversal
                else "Пробой подтверждён Perp CVD, OI и объёмом"
            ),
        ]

        if perp_confirms or (reversal and absorption):
            score += 1

            reasons.append(
                "Perp-поток подтверждает"
                if perp_confirms
                else "Возможное поглощение агрессивного потока"
            )

        if (
            (impulse and oi_buildup)
            or (reversal and oi_flush)
        ):
            score += 1

            reasons.append(
                "OI: рост открытых позиций"
                if oi_buildup
                else "OI: сокращение открытых позиций"
            )

        if divergence:
            score += 1
            reasons.append(
                "Дивергенция цены и Spot CVD"
            )

        if high_volume:
            score += 1
            reasons.append(
                "Объём выше среднего"
            )

        liquidation_confirms = (
            reversal_liquidations
            if reversal
            else impulse_liquidations
        )

        if liquidation_confirms:
            score += 1
            reasons.append(
                "Ликвидации аномальны относительно истории"
            )

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
            reasons.append(
                "L/S: контрнаправленный перекос"
            )

        if score < MIN_SCORE:
            rejected.append(
                f"{direction}: score={score}, "
                f"нужно {MIN_SCORE}"
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
            "spot_delta_pct": float(
                current["spot_delta_share"] * 100
            ),
            "perp_delta_pct": float(
                current["perp_delta_share"] * 100
            ),
            "spot_flow_3": float(spot_flow),
            "perp_flow_3": float(perp_flow),
            "long_liq": float(current["long_liq"]),
            "short_liq": float(current["short_liq"]),
            "ls_ratio": ratio,
            "fvg": bool(
                bullish_fvg if is_long else bearish_fvg
            ),
        })

    return candidates, "; ".join(rejected)


# ============================================================
# 9. СОХРАНЕНИЕ СОСТОЯНИЯ
# ============================================================

class StateStore:
    def __init__(self):
        self.db = sqlite3.connect(STATE_DB)

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id TEXT PRIMARY KEY,
                coin TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction TEXT NOT NULL,
                candle INTEGER NOT NULL,
                mode TEXT NOT NULL
            )
        """)

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS audit (
                created INTEGER NOT NULL,
                coin TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                candle INTEGER NOT NULL,
                result TEXT NOT NULL
            )
        """)

        self.db.commit()

    def record(self, coin, timeframe, candle, result):
        serialized = json.dumps(
            result,
            ensure_ascii=False,
        )

        self.db.execute(
            "INSERT INTO audit VALUES (?, ?, ?, ?, ?)",
            (
                int(time.time()),
                coin,
                timeframe,
                int(candle),
                serialized,
            ),
        )

        # Ограничение роста журнала.
        self.db.execute(
            "DELETE FROM audit WHERE created < ?",
            (int(time.time()) - 30 * 86400,),
        )

        self.db.commit()

        logger.info(
            "%s %s candle=%s | %s",
            coin,
            timeframe,
            candle,
            serialized,
        )

    def signal_id(
        self, coin, timeframe, direction, candle
    ):
        mode = "dry" if DRY_RUN else "live"

        return (
            f"{coin}:{timeframe}:{direction}:"
            f"{candle}:{mode}"
        )

    def already_sent(
        self, coin, timeframe, direction, candle
    ):
        signal_id = self.signal_id(
            coin, timeframe, direction, candle
        )

        row = self.db.execute(
            "SELECT 1 FROM signals WHERE id=?",
            (signal_id,),
        ).fetchone()

        return row is not None

    def cooldown_active(
        self, coin, timeframe, direction, candle
    ):
        mode = "dry" if DRY_RUN else "live"

        row = self.db.execute(
            """
            SELECT MAX(candle)
            FROM signals
            WHERE coin=?
              AND timeframe=?
              AND direction=?
              AND mode=?
            """,
            (coin, timeframe, direction, mode),
        ).fetchone()

        previous_candle = row[0]

        if previous_candle is None:
            return False

        seconds = TIMEFRAMES[timeframe]["seconds"]

        return (
            candle - previous_candle
            < COOLDOWN_BARS * seconds
        )

    def mark_sent(
        self, coin, timeframe, direction, candle
    ):
        mode = "dry" if DRY_RUN else "live"

        self.db.execute(
            """
            INSERT OR IGNORE INTO signals
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self.signal_id(
                    coin, timeframe, direction, candle
                ),
                coin,
                timeframe,
                direction,
                int(candle),
                mode,
            ),
        )

        self.db.commit()


# ============================================================
# 10. TELEGRAM
# ============================================================

def send_telegram(text):
    if DRY_RUN:
        logger.info("DRY_RUN message:
%s", text)
        return True

    try:
        response = requests.post(
            (
                "https://api.telegram.org/"
                f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            ),
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
            },
            timeout=20,
        )

        result = response.json()

        if (
            response.status_code == 200
            and result.get("ok") is True
        ):
            return True

        logger.error(
            "Telegram failed: HTTP=%s error_code=%s",
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
    coin,
    timeframe,
    candle,
    signal,
    btc_context,
    higher_context,
    market,
):
    seconds = TIMEFRAMES[timeframe]["seconds"]

    closed_at = datetime.fromtimestamp(
        candle + seconds,
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

    reasons = "
".join(
        f"• {html.escape(reason)}"
        for reason in signal["reasons"]
    )

    emoji = (
        "🟢"
        if signal["direction"] == "LONG"
        else "🔴"
    )

    spot_symbol = html.escape(
        market["spot"]["symbol"]
    )

    perp_symbol = html.escape(
        market["perp"]["symbol"]
    )

    return (
        f"{emoji} <b>ORDER FLOW | "
        f"{timeframe.upper()} {signal['direction']}</b>

"
        f"Монета: <b>#{html.escape(coin)}USDT</b>
"
        f"Цена закрытия: {signal['price']:,.4f}
"
        f"Свеча закрылась: {closed_at}
"
        f"Сетап: {html.escape(signal['setup'])}
"
        f"Баллы условий: {signal['score']}/10
"
        f"<i>Баллы — не вероятность успеха.</i>

"
        f"📊 Spot delta / volume: "
        f"{signal['spot_delta_pct']:+.1f}%
"
        f"📊 Perp delta / volume: "
        f"{signal['perp_delta_pct']:+.1f}%
"
        f"🏦 OI за 3 свечи: "
        f"{signal['oi_change']:+.2f}%
"
        f"💰 Funding: {signal['funding']:+.4f}%
"
        f"💥 Ликвидации LONG: "
        f"${signal['long_liq']:,.0f}
"
        f"💥 Ликвидации SHORT: "
        f"${signal['short_liq']:,.0f}
"
        f"⚖️ Long/Short Ratio: {ratio_text}
"
        f"📏 ATR(14): {signal['atr']:,.4f}
"
        f"📐 FVG по направлению: "
        f"{'да' if signal['fvg'] else 'нет'}

"
        f"BTC: {context_names.get(btc_context, btc_context)}
"
        f"Старший ТФ: "
        f"{context_names.get(higher_context, higher_context)}

"
        f"{reasons}

"
        f"Источник: Coinalyze
"
        f"Spot: {spot_symbol}
"
        f"Perp: {perp_symbol}
"
        f"<i>Один рынок каждого типа, не агрегат всех бирж. "
        f"Уведомление о сетапе, не команда на вход.</i>"
    )


# ============================================================
# 11. ПЛАНИРОВЩИК И ФИЛЬТРЫ КОНТЕКСТА
# ============================================================

def get_boundaries():
    # Один фиксированный срез времени на цикл.
    effective_time = (
        int(time.time()) - CLOSE_DELAY_SECONDS
    )

    return {
        timeframe: (
            effective_time // settings["seconds"]
        ) * settings["seconds"]
        for timeframe, settings in TIMEFRAMES.items()
    }


def context_block_reason(
    coin,
    timeframe,
    direction,
    cache,
    boundaries,
):
    opposite = (
        "bearish" if direction == "LONG" else "bullish"
    )

    btc_record = cache.get(("BTC", timeframe))

    if coin != "BTC":
        if (
            btc_record is None
            or btc_record["boundary"]
            != boundaries[timeframe]
        ):
            return "BTC-контекст отсутствует", None, None

        if btc_record["context"] == opposite:
            return (
                "BTC-контекст против сигнала",
                btc_record["context"],
                None,
            )

    higher_timeframe = HIGHER_TF.get(timeframe)

    higher_context = "not_used"

    if higher_timeframe:
        higher_record = cache.get(
            (coin, higher_timeframe)
        )

        if (
            higher_record is None
            or higher_record["boundary"]
            != boundaries[higher_timeframe]
        ):
            return (
                "Старший ТФ отсутствует",
                None,
                None,
            )

        higher_context = higher_record["context"]

        if higher_context == opposite:
            return (
                "Старший ТФ против сигнала",
                None,
                higher_context,
            )

    btc_context = (
        btc_record["context"]
        if btc_record is not None
        else "neutral"
    )

    return None, btc_context, higher_context


def main():
    if not COINALYZE_API_KEY:
        raise RuntimeError(
            "Задайте COINALYZE_API_KEY в Environment"
        )

    if not DRY_RUN and (
        not TELEGRAM_BOT_TOKEN
        or not TELEGRAM_CHAT_ID
    ):
        raise RuntimeError(
            "Задайте TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID"
        )

    if not 1 <= MIN_SCORE <= 10:
        raise RuntimeError(
            "MIN_SCORE должен быть от 1 до 10"
        )

    threading.Thread(
        target=start_health_server,
        daemon=True,
    ).start()

    client = CoinalyzeClient()
    state = StateStore()

    # Справочники загружаются один раз при старте.
    while True:
        try:
            market_map = build_market_map(client)
            break

        except AuthenticationError:
            raise

        except DataError as exc:
            HEALTH["last_error"] = str(exc)
            logger.error(
                "Market mapping failed: %s. Retry in 120s",
                str(exc),
            )
            time.sleep(120)

    cache = {}
    data_retry_after = {}
    telegram_retry_after = {}
    processed = set()

    mode = (
        "DRY RUN: сигналы только в логах"
        if DRY_RUN
        else "LIVE: отправка в Telegram"
    )

    send_telegram(
        "🚀 <b>Coinalyze Order Flow запущен</b>
"
        f"Монеты: {', '.join(WATCHLIST)}
"
        "Таймфреймы: 15m, 1h, 4h
"
        f"Режим: {mode}
"
        "Первичная загрузка может занять несколько минут."
    )

    HEALTH["status"] = "running"

    while True:
        boundaries = get_boundaries()

        # Сначала старшие ТФ, затем младшие.
        # BTC первым в WATCHLIST.
        for timeframe in LOAD_ORDER:
            boundary = boundaries[timeframe]

            for coin in WATCHLIST:
                key = (coin, timeframe)
                existing = cache.get(key)

                if (
                    existing is not None
                    and existing["boundary"] == boundary
                ):
                    continue

                if (
                    time.time()
                    < data_retry_after.get(key, 0)
                ):
                    continue

                try:
                    frame = load_market_data(
                        client,
                        market_map[coin],
                        timeframe,
                        boundary,
                    )

                    cache[key] = {
                        "boundary": boundary,
                        "frame": frame,
                        "context": get_context(frame),
                    }

                    data_retry_after.pop(key, None)

                    HEALTH["last_data_success"] = int(
                        time.time()
                    )

                    logger.info(
                        "DATA OK | %s %s | bars=%s | "
                        "context=%s",
                        coin,
                        timeframe,
                        len(frame),
                        cache[key]["context"],
                    )

                except AuthenticationError:
                    raise

                except DataError as exc:
                    cache.pop(key, None)

                    data_retry_after[key] = (
                        time.time() + DATA_RETRY_SECONDS
                    )

                    HEALTH["last_error"] = str(exc)

                    state.record(
                        coin,
                        timeframe,
                        boundary,
                        {"data_error": str(exc)},
                    )

        # Анализ после обновления контекстов.
        for timeframe in LOAD_ORDER:
            seconds = TIMEFRAMES[timeframe]["seconds"]

            for coin in WATCHLIST:
                record = cache.get((coin, timeframe))

                if (
                    record is None
                    or record["boundary"]
                    != boundaries[timeframe]
                ):
                    continue

                frame = record["frame"]
                candle = int(frame.index[-1])

                work_key = (coin, timeframe, candle)

                if work_key in processed:
                    continue

                candle_closed_at = candle + seconds

                if (
                    time.time() - candle_closed_at
                    > MAX_SIGNAL_DELAY_SECONDS
                ):
                    state.record(
                        coin,
                        timeframe,
                        candle,
                        {
                            "skip": (
                                "Сетап устарел для отправки; "
                                "данные используются как контекст"
                            )
                        },
                    )

                    processed.add(work_key)
                    continue

                candidates, explanation = (
                    analyze_orderflow(frame)
                )

                if not candidates:
                    state.record(
                        coin,
                        timeframe,
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

                # Защита от неоднозначного результата.
                if len(candidates) > 1:
                    state.record(
                        coin,
                        timeframe,
                        candle,
                        {
                            "skip": (
                                "Противоречивые направления"
                            )
                        },
                    )

                    processed.add(work_key)
                    continue

                signal = candidates[0]
                direction = signal["direction"]

                block, btc_context, higher_context = (
                    context_block_reason(
                        coin,
                        timeframe,
                        direction,
                        cache,
                        boundaries,
                    )
                )

                if block:
                    # Если контекст ещё не загрузился,
                    # можно повторить анализ в следующем цикле.
                    waiting_for_data = (
                        "отсутствует" in block
                    )

                    state.record(
                        coin,
                        timeframe,
                        candle,
                        {
                            "skip": block,
                            "candidate": signal,
                        },
                    )

                    if not waiting_for_data:
                        processed.add(work_key)

                    continue

                if state.already_sent(
                    coin,
                    timeframe,
                    direction,
                    candle,
                ):
                    processed.add(work_key)
                    continue

                if state.cooldown_active(
                    coin,
                    timeframe,
                    direction,
                    candle,
                ):
                    state.record(
                        coin,
                        timeframe,
                        candle,
                        {"skip": "Пауза между сигналами"},
                    )

                    processed.add(work_key)
                    continue

                if (
                    time.time()
                    < telegram_retry_after.get(
                        work_key, 0
                    )
                ):
                    continue

                message = format_signal(
                    coin,
                    timeframe,
                    candle,
                    signal,
                    btc_context,
                    higher_context,
                    market_map[coin],
                )

                if send_telegram(message):
                    state.mark_sent(
                        coin,
                        timeframe,
                        direction,
                        candle,
                    )

                    state.record(
                        coin,
                        timeframe,
                        candle,
                        {
                            "signal": signal,
                            "mode": (
                                "dry_run"
                                if DRY_RUN
                                else "telegram"
                            ),
                        },
                    )

                    processed.add(work_key)
                    telegram_retry_after.pop(
                        work_key, None
                    )

                else:
                    telegram_retry_after[work_key] = (
                        time.time()
                        + TELEGRAM_RETRY_SECONDS
                    )

        # Ограничиваем рост оперативной памяти.
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
        # Не выводим тексты сетевых исключений с URL.
        HEALTH["status"] = "fatal_error"
        HEALTH["last_error"] = type(exc).__name__

        logger.critical(
            "Fatal error: %s",
            type(exc).__name__,
        )

        raise SystemExit(1)
