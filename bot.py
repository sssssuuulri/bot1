#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import ccxt
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Для работы на сервере без GUI
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# ===================== НАСТРОЙКИ =====================
TELEGRAM_BOT_TOKEN = "8462678220:AAGYlYEpKbOp5Bt-1IVectAdlfIUZ2Bs73s"
CHAT_ID = "911511438"

TIMEFRAME = '1m'
HTF_TIMEFRAME = '15m'

MAX_SYMBOLS = 50  # Уменьшено для скорости
POLL_INTERVAL = 30
SIGNAL_COOLDOWN_MIN = 5  # Уменьшен кулдаун

MIN_VOLUME = 50000  # Немного увеличен
LOOKBACK_BARS = 150  # Уменьшено для скорости

CHART_DIR = "charts"
LOG_FILE = "bot.log"
os.makedirs(CHART_DIR, exist_ok=True)

# ===================== ЛОГИРОВАНИЕ =====================
def log_message(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}\n"
    print(log_entry.strip())
    
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except:
        pass

# ===================== TELEGRAM =====================
def send_telegram(text, image_path=None):
    try:
        if image_path and os.path.exists(image_path):
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path, "rb") as photo:
                files = {"photo": photo}
                data = {"chat_id": CHAT_ID, "caption": text, "parse_mode": "HTML"}
                response = requests.post(url, data=data, files=files, timeout=30)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
            response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            return True
        else:
            log_message(f"Telegram ошибка: {response.status_code} - {response.text}", "ERROR")
            return False
            
    except Exception as e:
        log_message(f"Ошибка отправки в Telegram: {e}", "ERROR")
        return False

# ===================== ИНДИКАТОРЫ =====================
def calculate_rsi(prices, period=14):
    """Расчет RSI"""
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    
    if down == 0:
        return 100.0
    
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100.0 - 100.0 / (1.0 + rs)
    
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        
        if delta > 0:
            up_val = delta
            down_val = 0.0
        else:
            up_val = 0.0
            down_val = -delta
        
        up = (up * (period - 1) + up_val) / period
        down = (down * (period - 1) + down_val) / period
        
        if down == 0:
            rsi[i] = 100.0
        else:
            rs = up / down
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    
    return rsi

def calculate_ema(prices, period):
    """Расчет EMA"""
    return pd.Series(prices).ewm(span=period, adjust=False).mean().values

def calculate_bb(prices, period=20, mult=2.0):
    """Расчет Bollinger Bands"""
    basis = pd.Series(prices).rolling(window=period).mean().values
    std = pd.Series(prices).rolling(window=period).std().values
    upper = basis + (std * mult)
    lower = basis - (std * mult)
    return basis, upper, lower

def check_signal(df, htf_ema_value=None):
    """
    Проверка сигналов по стратегии TradingView
    Возвращает: (long_signal, short_signal, indicators_dict)
    """
    try:
        # Параметры
        LEN_RSI = 14
        LEN_EMA = 50
        LEN_BB = 20
        BB_MULT = 1.8
        THR_FOMO_UP = 65
        THR_PANIC = 35
        USE_HTF = htf_ema_value is not None
        
        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        
        # Расчет индикаторов
        rsi_values = calculate_rsi(closes, LEN_RSI)
        ema_values = calculate_ema(closes, LEN_EMA)
        bb_basis, bb_upper, bb_lower = calculate_bb(closes, LEN_BB, BB_MULT)
        
        # Текущие значения (последний бар)
        current_idx = len(closes) - 1
        current_close = closes[current_idx]
        current_open = opens[current_idx]
        current_high = highs[current_idx]
        current_low = lows[current_idx]
        current_volume = volumes[current_idx]
        current_rsi = rsi_values[current_idx]
        current_ema = ema_values[current_idx]
        current_bb_upper = bb_upper[current_idx]
        current_bb_lower = bb_lower[current_idx]
        
        # Volume Z-score
        volume_mean = np.mean(volumes[-LEN_BB:])
        volume_std = np.std(volumes[-LEN_BB:])
        if volume_std > 0:
            vol_z = (current_volume - volume_mean) / volume_std
        else:
            vol_z = 0
        
        # Условия для LONG
        long_conditions = []
        
        # 1. RSI пересек снизу вверх уровень 35
        if current_idx > 0:
            rsi_cross_up = (rsi_values[current_idx-1] < THR_PANIC) and (current_rsi > THR_PANIC)
            long_conditions.append(("RSI cross up", rsi_cross_up))
        
        # 2. Цена вернулась внутрь BB снизу
        bb_return_up = False
        if current_idx > 0:
            bb_return_up = (closes[current_idx-1] <= bb_lower[current_idx-1]) and (current_close > current_bb_lower)
        long_conditions.append(("BB return up", bb_return_up))
        
        # 3. Бычья свеча
        bull_candle = current_close > current_open
        candle_range = current_high - current_low
        if candle_range > 0:
            body_pct = abs(current_close - current_open) / candle_range
            strong_bull = bull_candle and (body_pct >= 0.45)
        else:
            strong_bull = False
        long_conditions.append(("Bull candle", strong_bull))
        
        # 4. Цена выше EMA
        above_ema = current_close >= current_ema
        long_conditions.append(("Above EMA", above_ema))
        
        # 5. Объем выше среднего
        good_volume = vol_z >= -0.5
        long_conditions.append(("Good volume", good_volume))
        
        # 6. HTF подтверждение
        htf_confirm = True
        if USE_HTF:
            htf_confirm = current_close >= htf_ema_value
        long_conditions.append(("HTF confirm", htf_confirm))
        
        # Условия для SHORT
        short_conditions = []
        
        # 1. RSI пересек сверху вниз уровень 65
        if current_idx > 0:
            rsi_cross_down = (rsi_values[current_idx-1] > THR_FOMO_UP) and (current_rsi < THR_FOMO_UP)
            short_conditions.append(("RSI cross down", rsi_cross_down))
        
        # 2. Цена вернулась внутрь BB сверху
        bb_return_down = False
        if current_idx > 0:
            bb_return_down = (closes[current_idx-1] >= bb_upper[current_idx-1]) and (current_close < current_bb_upper)
        short_conditions.append(("BB return down", bb_return_down))
        
        # 3. Медвежья свеча
        bear_candle = current_close < current_open
        if candle_range > 0:
            body_pct = abs(current_close - current_open) / candle_range
            strong_bear = bear_candle and (body_pct >= 0.45)
        else:
            strong_bear = False
        short_conditions.append(("Bear candle", strong_bear))
        
        # 4. Цена ниже EMA
        below_ema = current_close <= current_ema
        short_conditions.append(("Below EMA", below_ema))
        
        # 5. Объем выше среднего
        short_conditions.append(("Good volume", good_volume))
        
        # 6. HTF подтверждение
        htf_confirm_short = True
        if USE_HTF:
            htf_confirm_short = current_close <= htf_ema_value
        short_conditions.append(("HTF confirm", htf_confirm_short))
        
        # Проверяем все условия
        long_signal = all(cond[1] for cond in long_conditions)
        short_signal = all(cond[1] for cond in short_conditions)
        
        indicators = {
            'price': float(current_close),
            'rsi': float(current_rsi),
            'ema': float(current_ema),
            'bb_upper': float(current_bb_upper),
            'bb_lower': float(current_bb_lower),
            'volume': float(current_volume),
            'vol_z': float(vol_z),
            'long_conditions': long_conditions,
            'short_conditions': short_conditions
        }
        
        return long_signal, short_signal, indicators
        
    except Exception as e:
        log_message(f"Ошибка в check_signal: {e}", "ERROR")
        return False, False, {}

# ===================== ГРАФИК =====================
def create_chart(df, symbol, indicators, signal_type):
    """Создание графика для сигнала"""
    try:
        plt.figure(figsize=(12, 8))
        
        # Цена и индикаторы
        plt.subplot(3, 1, 1)
        closes = df['close'].values[-100:]  # Последние 100 баров
        plt.plot(closes, label='Цена', color='black', linewidth=1)
        
        # Добавляем индикаторы если есть
        if 'ema' in indicators:
            ema_vals = calculate_ema(df['close'].values, 50)[-100:]
            plt.plot(ema_vals, label='EMA 50', color='orange', linewidth=2, alpha=0.7)
        
        # Сигнальная точка
        color = 'green' if signal_type == 'LONG' else 'red'
        marker = '^' if signal_type == 'LONG' else 'v'
        plt.scatter(len(closes)-1, closes[-1], color=color, s=200, 
                   marker=marker, label=f'{signal_type} Signal', zorder=5)
        
        plt.title(f"{symbol} - {signal_type} Signal", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # RSI
        plt.subplot(3, 1, 2)
        rsi_vals = calculate_rsi(df['close'].values, 14)[-100:]
        plt.plot(rsi_vals, label='RSI 14', color='purple', linewidth=1)
        plt.axhline(y=65, color='red', linestyle='--', alpha=0.5, label='FOMO Up (65)')
        plt.axhline(y=35, color='green', linestyle='--', alpha=0.5, label='Panic (35)')
        plt.ylim(0, 100)
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Объем
        plt.subplot(3, 1, 3)
        volumes = df['volume'].values[-100:]
        colors = ['green' if df['close'].iloc[i] > df['open'].iloc[i] else 'red' 
                 for i in range(-100, 0)]
        plt.bar(range(len(volumes)), volumes, color=colors, alpha=0.7)
        plt.ylabel('Объем')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохранение
        timestamp = int(time.time())
        filename = f"{CHART_DIR}/{symbol.replace('/', '_')}_{signal_type}_{timestamp}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        return filename
        
    except Exception as e:
        log_message(f"Ошибка создания графика: {e}", "ERROR")
        return None

# ===================== ОСНОВНОЙ КОД =====================
def main():
    """Основная функция бота"""
    log_message("=" * 60)
    log_message("🚀 ЗАПУСК ТОРГОВОГО БОТА")
    log_message("=" * 60)
    
    # Стартовое сообщение
    start_msg = f"""<b>🤖 Бот запущен</b>

📊 <b>Конфигурация:</b>
• Биржа: Bybit Futures
• Таймфрейм: 1m
• HTF подтверждение: 15m
• Макс. пар: {MAX_SYMBOLS}
• Интервал: {POLL_INTERVAL} сек
• Кулдаун: {SIGNAL_COOLDOWN_MIN} мин

⏰ <b>Время запуска:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    send_telegram(start_msg)
    
    # Инициализация биржи
    try:
        exchange = ccxt.bybit({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
            "timeout": 30000
        })
        
        # Проверка подключения
        exchange.fetch_time()
        log_message("✅ Подключение к Bybit установлено")
        
    except Exception as e:
        log_message(f"❌ Ошибка подключения к Bybit: {e}", "ERROR")
        send_telegram(f"<b>❌ Ошибка подключения к Bybit:</b>\n{e}")
        return
    
    # Загрузка торговых пар
    try:
        markets = exchange.load_markets()
        symbols = [
            s for s, m in markets.items()
            if m.get('swap') and m.get('linear') and 
            s.endswith('USDT') and not '1000' in s
        ]
        symbols = sorted(symbols)[:MAX_SYMBOLS]
        log_message(f"✅ Загружено {len(symbols)} торговых пар")
        
    except Exception as e:
        log_message(f"❌ Ошибка загрузки торговых пар: {e}", "ERROR")
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']  # Fallback
    
    # Основной цикл
    cycle_count = 0
    signal_count = 0
    recent_signals = {}
    
    while True:
        try:
            cycle_count += 1
            log_message(f"\n🔍 Цикл #{cycle_count} - Проверка {len(symbols)} пар")
            
            for symbol in symbols:
                try:
                    # Проверка кулдауна
                    now = time.time()
                    if symbol in recent_signals:
                        if now - recent_signals[symbol] < SIGNAL_COOLDOWN_MIN * 60:
                            continue
                    
                    # Загрузка данных
                    ohlcv_1m = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=LOOKBACK_BARS + 20)
                    ohlcv_15m = exchange.fetch_ohlcv(symbol, HTF_TIMEFRAME, limit=50)
                    
                    if len(ohlcv_1m) < LOOKBACK_BARS:
                        continue
                    
                    # Создание DataFrame
                    df_1m = pd.DataFrame(
                        ohlcv_1m[-LOOKBACK_BARS:],
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    
                    # Расчет HTF EMA
                    htf_ema = None
                    if len(ohlcv_15m) > 20:
                        df_15m = pd.DataFrame(
                            ohlcv_15m,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )
                        htf_ema = float(df_15m['close'].ewm(span=50, adjust=False).mean().iloc[-1])
                    
                    # Проверка объема
                    current_volume = df_1m['volume'].iloc[-1]
                    if current_volume < MIN_VOLUME:
                        continue
                    
                    # Проверка сигнала
                    long_signal, short_signal, indicators = check_signal(df_1m, htf_ema)
                    
                    if long_signal or short_signal:
                        signal_type = "LONG" if long_signal else "SHORT"
                        recent_signals[symbol] = now
                        signal_count += 1
                        
                        # Создание графика
                        chart_file = create_chart(df_1m, symbol, indicators, signal_type)
                        
                        # Формирование сообщения
                        emoji = "🟢" if long_signal else "🔴"
                        signal_emoji = "🚀" if long_signal else "📉"
                        
                        # Форматирование условий
                        conditions = indicators.get('long_conditions' if long_signal else 'short_conditions', [])
                        conditions_text = "\n".join([f"• {name}: {'✅' if value else '❌'}" 
                                                   for name, value in conditions])
                        
                        message = f"""{emoji} <b>{signal_emoji} СИГНАЛ {signal_type} {signal_emoji}</b> {emoji}

<b>🎯 Пара:</b> {symbol}
<b>💰 Цена:</b> ${indicators['price']:.4f}
<b>📊 RSI:</b> {indicators['rsi']:.1f}
<b>📈 Объем (Z-score):</b> {indicators['vol_z']:.2f}

<b>📊 Индикаторы:</b>
• EMA 50: ${indicators['ema']:.4f}
• BB Верх: ${indicators['bb_upper']:.4f}
• BB Низ: ${indicators['bb_lower']:.4f}
• Объем: {int(indicators['volume']):,}

<b>✅ Условия:</b>
{conditions_text}

<b>⏰ Время:</b> {datetime.now().strftime('%H:%M:%S')}
<b>🔢 Сигналов сегодня:</b> {signal_count}
"""
                        
                        # Отправка
                        log_message(f"📢 Найден сигнал {signal_type} для {symbol}")
                        send_telegram(message, chart_file)
                        
                        # Очистка старого графика
                        if chart_file and os.path.exists(chart_file):
                            time.sleep(5)  # Даем время на отправку
                            try:
                                os.remove(chart_file)
                            except:
                                pass
                        
                        # Пауза между сигналами
                        time.sleep(2)
                    
                except ccxt.NetworkError as e:
                    log_message(f"Сетевая ошибка для {symbol}: {e}", "WARNING")
                    time.sleep(5)
                except ccxt.ExchangeError as e:
                    log_message(f"Ошибка биржи для {symbol}: {e}", "WARNING")
                    time.sleep(3)
                except Exception as e:
                    log_message(f"Ошибка обработки {symbol}: {type(e).__name__}: {e}", "ERROR")
                    continue
            
            # Статус
            if cycle_count % 10 == 0:
                status_msg = f"""<b>📊 Статус бота</b>

✅ Циклов выполнено: {cycle_count}
📈 Найдено сигналов: {signal_count}
🔍 Активных пар: {len(symbols)}
⏰ Следующая проверка: через {POLL_INTERVAL} сек
"""
                send_telegram(status_msg)
            
            log_message(f"✅ Цикл #{cycle_count} завершен. Ожидание {POLL_INTERVAL} сек...")
            time.sleep(POLL_INTERVAL)
            
        except KeyboardInterrupt:
            log_message("\n👋 Остановка по команде пользователя")
            break
        except Exception as e:
            log_message(f"❌ Критическая ошибка в основном цикле: {e}", "ERROR")
            time.sleep(30)

    # Завершение работы
    end_msg = f"""<b>🛑 Бот остановлен</b>

📊 <b>Итоги работы:</b>
• Циклов выполнено: {cycle_count}
• Сигналов найдено: {signal_count}
• Время работы: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    send_telegram(end_msg)
    log_message("=" * 60)
    log_message("БОТ ОСТАНОВЛЕН")
    log_message("=" * 60)

if __name__ == "__main__":
    main()
