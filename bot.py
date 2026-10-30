import os
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# =====================================================================
# 0. ВЕБ-СЕРВЕР ДЛЯ ПОДДЕРЖАНИЯ RENDER LIVE
# =====================================================================
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running 24/7!")
    def log_message(self, format, *args):
        pass

def start_web_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

# =====================================================================
# 1. НАСТРОЙКИ TELEGRAM И СПИСОК 60 МОНЕТ
# =====================================================================
TELEGRAM_BOT_TOKEN = "8506486802:AAFjFf_bi0f3KRJbKI9SsxPKr8ykqqWOTEs"
TELEGRAM_CHAT_ID = "53516056"

TOP_COINS_COUNT = 60
CLUSTER_ALERT_FAST = 5
CLUSTER_ALERT_HTF = 3

# Железобетонный резервный список из 60 топ-монет (на случай любых блокировок)
DEFAULT_TOP_60 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "TRXUSDT", "LINKUSDT",
    "DOTUSDT", "NEARUSDT", "MATICUSDT", "LTCUSDT", "BCHUSDT", "UNIUSDT", "ICPUSDT", "APTUSDT", "FETUSDT", "ETCUSDT",
    "RNDRUSDT", "ATOMUSDT", "STXUSDT", "FILUSDT", "ARBUSDT", "IMXUSDT", "OPUSDT", "INJUSDT", "SUIUSDT", "TIAUSDT",
    "SEIUSDT", "MEMEUSDT", "ORDIUSDT", "WLDUSDT", "PEPEUSDT", "SHIBUSDT", "WIFUSDT", "BONKUSDT", "FLOKIUSDT", "GALAUSDT",
    "SANDUSDT", "MANAUSDT", "AXSUSDT", "APEUSDT", "LDOUSDT", "AAVEUSDT", "MKRUSDT", "SNXUSDT", "CRVUSDT", "DYDXUSDT",
    "RUNEUSDT", "FTMUSDT", "ALGOUSDT", "VETUSDT", "EGLDUSDT", "QNTUSDT", "THETAUSDT", "EOSUSDT", "XTZUSDT", "FLOWUSDT"
]

def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Ошибка Telegram: {e}")

# =====================================================================
# 2. ЗАГРУЗКА ТОП-60 МОНЕТ (МУЛЬТИ-ИСТОЧНИК БЕЗ ГЕО-БЛОКА)
# =====================================================================
def get_top_futures_symbols(limit=TOP_COINS_COUNT) -> list:
    # 1. Пробуем публичное зеркало Binance Vision (разрешено в США)
    try:
        resp = requests.get("https://data-api.binance.vision/api/v3/ticker/24hr", timeout=8).json()
        if isinstance(resp, list) and len(resp) > 50:
            ignore = {"USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "USDPUSDT", "EURUSDT", "DAIUSDT"}
            usdt_pairs = [
                item for item in resp 
                if item['symbol'].endswith('USDT') and item['symbol'] not in ignore
                and not item['symbol'].endswith(('UPUSDT', 'DOWNUSDT'))
            ]
            usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
            symbols = [item['symbol'] for item in usdt_pairs[:limit]]
            if len(symbols) >= limit:
                return symbols
    except Exception as e:
        print(f"Ошибка зеркала Binance Vision: {e}")

    # 2. Пробуем фьючерсы Bybit
    try:
        resp = requests.get("https://api.bybit.com/v5/market/tickers?category=linear", timeout=8).json()
        rows = resp.get("result", {}).get("list", [])
        if len(rows) > 50:
            usdt_pairs = [item for item in rows if item['symbol'].endswith('USDT') and item['symbol'] != 'USDCUSDT']
            usdt_pairs.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            symbols = [item['symbol'] for item in usdt_pairs[:limit]]
            if len(symbols) >= limit:
                return symbols
    except Exception as e:
        print(f"Ошибка топа Bybit: {e}")

    # 3. Если оба заблокированы — отдаем полный вшитый список из 60 монет!
    return DEFAULT_TOP_60[:limit]

# =====================================================================
# 3. УНИВЕРСАЛЬНАЯ ЗАГРУЗКА СВЕЧЕЙ (ОБХОД БЛОКИРОВКИ США)
# =====================================================================
def fetch_klines(symbol: str, interval: str, limit: int = 35):
    # Спотовый эндпоинт Binance Vision (работает из США и отдает точный Taker Buy Volume!)
    try:
        url = "https://data-api.binance.vision/api/v3/klines"
        data = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=5).json()
        if isinstance(data, list) and len(data) >= 15:
            df = pd.DataFrame(data, columns=['t', 'open', 'high', 'low', 'close', 'vol', 'ct', 'q_vol', 'trades', 'buy_vol', 'q_buy_vol', 'ignore'])
            for col in ['open', 'high', 'low', 'close', 'vol', 'buy_vol']:
                df[col] = df[col].astype(float)
            df['sell_vol'] = df['vol'] - df['buy_vol']
            df['delta'] = df['buy_vol'] - df['sell_vol']
            return df
    except Exception:
        pass

    # Резерв через фьючерсы Bybit (если монеты нет на споте Binance)
    try:
        tf_map = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
        url = "https://api.bybit.com/v5/market/kline"
        params = {"category": "linear", "symbol": symbol, "interval": tf_map.get(interval, "15"), "limit": limit}
        resp = requests.get(url, params=params, timeout=5).json()
        rows = resp.get("result", {}).get("list", [])
        if rows and len(rows) >= 15:
            rows.reverse()  # Bybit отдает от новых к старым, разворачиваем
            df = pd.DataFrame(rows, columns=['t', 'open', 'high', 'low', 'close', 'vol', 'turnover'])
            for col in ['open', 'high', 'low', 'close', 'vol']:
                df[col] = df[col].astype(float)
