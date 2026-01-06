#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import ccxt
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===================== НАСТРОЙКИ =====================
TELEGRAM_BOT_TOKEN = "8462678220:AAGYlYEpKbOp5Bt-1IVectAdlfIUZ2Bs73s"
CHAT_ID = 911511438

TIMEFRAME_FAST = '1m'
TIMEFRAME_SLOW = '1h'

MAX_SYMBOLS = 500
POLL_INTERVAL = 60
SIGNAL_COOLDOWN_MIN = 20

PRICE_CHANGE_THRESHOLD = 4.0
VOLUME_Z_THRESHOLD = 1.0
MIN_VOLUME = 30000

CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ===================== TELEGRAM =====================
def send_telegram(text, image=None):
    if image:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        with open(image, "rb") as f:
            requests.post(
                url,
                data={"chat_id": CHAT_ID, "caption": text},
                files={"photo": f},
                timeout=10
            )
    else:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(
            url,
            json={"chat_id": CHAT_ID, "text": text},
            timeout=10
        )

# ===================== ИНДИКАТОРЫ (БЕЗ pandas-ta) =====================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def fib_levels(high, low):
    d = high - low
    return {
        '0.236': high - d * 0.236,
        '0.382': high - d * 0.382,
        '0.5':   high - d * 0.5,
        '0.618': high - d * 0.618,
        '0.786': high - d * 0.786
    }

# ===================== ГРАФИК =====================
def build_chart(df, fib, symbol, tf):
    plt.figure(figsize=(10, 5))
    plt.plot(df['close'], label='Price')

    for price in fib.values():
        plt.hlines(price, xmin=0, xmax=len(df), linestyles='dashed')

    plt.title(f"{symbol} {tf}")
    plt.grid()

    filename = f"{CHART_DIR}/{symbol.replace('/', '')}_{tf}.png"
    plt.savefig(filename)
    plt.close()
    return filename

# ===================== ОСНОВНОЙ КОД =====================
def main():
    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
            "linear": True
        }
    })

    markets = exchange.load_markets()
    symbols = [
        s for s, m in markets.items()
        if m.get("swap") and m.get("linear") and s.endswith("USDT")
    ][:MAX_SYMBOLS]

    send_telegram(f"🚀 Бот запущен\nМонет: {len(symbols)}")
    recent = {}

    while True:
        for symbol in symbols:
            try:
                now = time.time()
                if symbol in recent and now - recent[symbol] < SIGNAL_COOLDOWN_MIN * 60:
                    continue

                ohlcv_1m = exchange.fetch_ohlcv(symbol, TIMEFRAME_FAST, limit=120)
                ohlcv_1h = exchange.fetch_ohlcv(symbol, TIMEFRAME_SLOW, limit=120)

                df_1m = pd.DataFrame(ohlcv_1m, columns=['t','open','high','low','close','volume'])
                df_1h = pd.DataFrame(ohlcv_1h, columns=['t','open','high','low','close','volume'])

                df_1m['RSI'] = rsi(df_1m['close'])
                df_1h['RSI'] = rsi(df_1h['close'])

                macd_line, signal_line = macd(df_1m['close'])

                price_change = (
                    (df_1m['close'].iloc[-1] - df_1m['close'].iloc[-20])
                    / df_1m['close'].iloc[-20] * 100
                )

                vol_z = (
                    (df_1m['volume'].iloc[-1] - df_1m['volume'].mean())
                    / df_1m['volume'].std()
                )

                if abs(price_change) < PRICE_CHANGE_THRESHOLD:
                    continue
                if vol_z < VOLUME_Z_THRESHOLD:
                    continue
                if df_1m['volume'].iloc[-1] < MIN_VOLUME:
                    continue

                recent[symbol] = now

                fib = fib_levels(df_1m['high'].max(), df_1m['low'].min())
                chart = build_chart(df_1m, fib, symbol, TIMEFRAME_FAST)

                msg = (
                    f"🚨 {symbol}\n"
                    f"Pump: {price_change:+.2f}%\n"
                    f"RSI 1M: {df_1m['RSI'].iloc[-1]:.1f}\n"
                    f"RSI 1H: {df_1h['RSI'].iloc[-1]:.1f}\n"
                    f"MACD: {macd_line.iloc[-1]:.4f}\n"
                    f"Биржа: Bybit Futures"
                )

                send_telegram(msg, chart)
                print(msg)

            except Exception as e:
                print("ERR:", e)
                continue

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
