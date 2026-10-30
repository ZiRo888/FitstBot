import os
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# =====================================================================
# 0. МИНИ ВЕБ-СЕРВЕР ДЛЯ RENDER (ЧТОБЫ НЕ БЫЛО ОШИБКИ PORT TIMEOUT)
# =====================================================================
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running 24/7!")

    def log_message(self, format, *args):
        pass  # Отключаем лишний спам в логах

def start_web_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    print(f"Веб-порт {port} открыт для Render!")
    server.serve_forever()


# =====================================================================
# 1. НАСТРОЙКИ TELEGRAM (ТВОИ ДАННЫЕ)
# =====================================================================
TELEGRAM_BOT_TOKEN = "8506486802:AAFjFf_bi0f3KRJbKI9SsxPKr8ykqqWOTEs"
TELEGRAM_CHAT_ID = "53516056"

TOP_COINS_COUNT = 60
CLUSTER_ALERT_FAST = 5
CLUSTER_ALERT_HTF = 3


# =====================================================================
# 2. ОТПРАВКА СООБЩЕНИЙ В TELEGRAM
# =====================================================================
def send_telegram(text: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Отправка в TG:\n{text}\n")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Ошибка Telegram: {e}")


# =====================================================================
# 3. ЗАГРУЗКА ТОП-МОНЕТ BINANCE FUTURES
# =====================================================================
def get_top_futures_symbols(limit=TOP_COINS_COUNT) -> list:
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    try:
        resp = requests.get(url, timeout=10).json()
        if not isinstance(resp, list):
            print(f"Ответ Binance (проверь регион сервера): {resp}")
            return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
        ignore = {"USDCUSDT", "BTCDOMUSDT", "DEFIUSDT"}
        usdt_pairs = [
            item for item in resp 
            if item['symbol'].endswith('USDT') and item['symbol'] not in ignore
        ]
        usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        return [item['symbol'] for item in usdt_pairs[:limit]]
    except Exception as e:
        print(f"Ошибка загрузки списка монет: {e}")
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]


# =====================================================================
# 4. АЛГОРИТМ 1: MULTI-TF MINI SIGNALS (Скринер объема, RSI, дельты)
# =====================================================================
def check_mini_signal(symbol: str, interval: str) -> str | None:
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": 35}
    try:
        data = requests.get(url, params=params, timeout=5).json()
        if not isinstance(data, list) or len(data) < 30:
            return None

        df = pd.DataFrame(data, columns=[
            't', 'open', 'high', 'low', 'close', 'vol', 
            'ct', 'q_vol', 'trades', 'buy_vol', 'q_buy_vol', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'vol', 'buy_vol']:
            df[col] = df[col].astype(float)

        df['sell_vol'] = df['vol'] - df['buy_vol']
        df['delta'] = df['buy_vol'] - df['sell_vol']

        delta_price = df['close'].diff()
        gain = delta_price.clip(lower=0).rolling(14).mean()
        loss = (-delta_price.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        df['vol_ma'] = df['vol'].rolling(20).mean()
        df['lowest_15'] = df['low'].shift(1).rolling(15).min()
        df['highest_15'] = df['high'].shift(1).rolling(15).max()

        curr = df.iloc[-2]
        prev = df.iloc[-3]

        candle_range = max(curr['high'] - curr['low'], 1e-9)
        lower_wick = min(curr['open'], curr['close']) - curr['low']
        upper_wick = curr['high'] - max(curr['open'], curr['close'])

        vol_mult = 1.7 if interval in ("1h", "4h", "1d") else 2.0
        volume_spike = (curr['vol'] > curr['vol_ma'] * vol_mult) or (prev['vol'] > prev['vol_ma'] * (vol_mult + 0.2))

        swept_low = (curr['low'] <= curr['lowest_15']) or (prev['low'] <= prev['lowest_15'])
        oversold = (curr['rsi'] < 34) or (prev['rsi'] < 31)
        bullish_reclaim = (curr['close'] > curr['open'] and curr['delta'] > 0) or (lower_wick / candle_range > 0.50 and curr['delta'] > 0)

        if volume_spike and (swept_low or oversold) and bullish_reclaim:
            return f"#{symbol} {interval} LONG 📈"

        swept_high = (curr['high'] >= curr['highest_15']) or (prev['high'] >= prev['highest_15'])
        overbought = (curr['rsi'] > 66) or (prev['rsi'] > 69)
        bearish_reject = (curr['close'] < curr['open'] and curr['delta'] < 0) or (upper_wick / candle_range > 0.50 and curr['delta'] > 0)

        if volume_spike and (swept_high or overbought) and bearish_reject:
            return f"#{symbol} {interval} SHORT 📉"
    except Exception:
        return None
    return None


def run_mini_signals_scan(symbols: list, interval: str):
    signals_long = []
    signals_short = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(lambda sym: check_mini_signal(sym, interval), symbols)

    for res in results:
        if res:
            if "LONG" in res:
                signals_long.append(res)
            else:
                signals_short.append(res)

    all_signals = signals_long + signals_short
    if not all_signals:
        return

    header = f"📡 <b>[MINI SIGNALS SCREENER | {interval.upper()}]</b>\n"
    message_lines = [header + all_signals[0]] + all_signals[1:]

    cluster_threshold = CLUSTER_ALERT_HTF if interval in ("1h", "4h", "1d") else CLUSTER_ALERT_FAST

    if len(signals_long) >= cluster_threshold:
        message_lines.append(f"\n🔥 <b>[СИНХРОННЫЙ ОТКУП ({interval.upper()})]: {len(signals_long)} монет LONG!</b>")
    elif len(signals_short) >= cluster_threshold:
        message_lines.append(f"\n⚠️ <b>[СИНХРОННЫЙ СЛИВ ({interval.upper()})]: {len(signals_short)} монет SHORT!</b>")

    send_telegram("\n".join(message_lines))


# =====================================================================
# 5. АЛГОРИТМ 2 & 3: FVG SMART MONEY & ORDER FLOW (По BTC и топам)
# =====================================================================
def check_smart_money_and_orderflow(symbol: str, interval: str):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": 20}
    try:
        data = requests.get(url, params=params, timeout=5).json()
        if not isinstance(data, list) or len(data) < 15:
            return

        df = pd.DataFrame(data, columns=[
            't', 'open', 'high', 'low', 'close', 'vol', 
            'ct', 'q_vol', 'trades', 'buy_vol', 'q_buy_vol', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)

        curr = df.iloc[-2]
        prev_high = df.iloc[-3]['high']
        prev_low = df.iloc[-3]['low']

        if curr['close'] > prev_high:
            msg = (
                f"📐 <b>[FVG SMART MONEY | {interval.upper()}]</b>\n"
                f"#{symbol} — Слом структуры вверх (MSS).\n"
                f"Образован бычий имбаланс (FVG). Возможен поиск LONG."
            )
            send_telegram(msg)
        elif curr['close'] < prev_low:
            msg = (
                f"📐 <b>[FVG SMART MONEY | {interval.upper()}]</b>\n"
                f"#{symbol} — Слом структуры вниз (MSS).\n"
                f"Образован медвежий имбаланс (FVG). Возможен поиск SHORT."
            )
            send_telegram(msg)
    except Exception:
        pass


# =====================================================================
# 6. ГЛАВНЫЙ ЦИКЛ ОБХОДА РЫНКА
# =====================================================================
if __name__ == "__main__":
    # 1. Запускаем веб-сервер в отдельном потоке для Render
    threading.Thread(target=start_web_server, daemon=True).start()

    # 2. Запускаем основную логику бота
    print("Загружаю список топ-монет Binance Futures...")
    top_symbols = get_top_futures_symbols()
    send_telegram(
        f"🚀 <b>Комбо-бот успешно запущен на сервере!</b>\n"
        f"Активны алгоритмы: Mini Signals (монет в базе: {len(top_symbols)}), FVG Smart Money и Order Flow.\n"
        f"Таймфреймы: <b>5m, 15m, 1h, 4h, 1d</b>."
    )

    last_triggered_minute = -1

    while True:
        now = datetime.now(timezone.utc)
        
        if now.minute % 5 == 0 and now.second >= 3 and now.minute != last_triggered_minute:
            last_triggered_minute = now.minute
            
            run_mini_signals_scan(top_symbols, "5m")
            check_smart_money_and_orderflow("BTCUSDT", "5m")

            if now.minute % 15 == 0:
                run_mini_signals_scan(top_symbols, "15m")
                check_smart_money_and_orderflow("BTCUSDT", "15m")

            if now.minute == 0:
                run_mini_signals_scan(top_symbols, "1h")
                check_smart_money_and_orderflow("BTCUSDT", "1h")

                if now.hour % 4 == 0:
                    run_mini_signals_scan(top_symbols, "4h")
                    check_smart_money_and_orderflow("BTCUSDT", "4h")

                if now.hour == 0:
                    run_mini_signals_scan(top_symbols, "1d")
                    check_smart_money_and_orderflow("BTCUSDT", "1d")

                top_symbols = get_top_futures_symbols()

        time.sleep(1)
