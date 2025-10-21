
import requests, pandas as pd, numpy as np, time, os
from datetime import datetime
from dotenv import load_dotenv
import schedule

# === LOAD ENV VARIABLES ===
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or BOT_TOKEN
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or CHAT_ID
TELEGRAM_ENABLED = True  # bật/tắt telegram

# ==== SYMBOLS & TIMEFRAMES ====
SYMBOLS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT"]
TIMEFRAMES = {"15m":15, "4h":240, "1d":1440, "1w":10080}
LIMIT = 250

# ==== 1. TELEGRAM SEND ====
def send_telegram(message: str):
    if not TELEGRAM_ENABLED: return
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token/chat not configured")
        return
    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode':'Markdown'}
        resp = requests.post(url, data=payload, timeout=10)
        if not resp.ok: print("Telegram send failed:", resp.text)
    except Exception as e:
        print("send_telegram error:", e)

def send_start_message():
    msg = f"🤖 Bot started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nBot đang chạy và sẽ gửi báo cáo/trade signal theo lịch."
    send_telegram(msg)

# ==== 2. FETCH KLINES ====
def fetch_klines(symbol, tf):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={tf}&limit={LIMIT}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['t','o','h','l','c','v','ct','q','n','tb','tq','ig'])
    df = df[['t','o','h','l','c','v']].astype(float)
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    return df

# ==== 3. INDICATORS ====
def atr(df, period=14):
    if len(df) < (period + 1):
        return []
    high_low = df['h']-df['l']
    high_close = abs(df['h']-df['c'].shift())
    low_close = abs(df['l']-df['c'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def ema(df, period):
    if len(df) < (period + 1):
        return []
    return df['c'].ewm(span=period, adjust=False).mean()

def rsi(df, period=14):
    if len(df) < (period + 1):
        return []
    delta = df['c'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up/ma_down
    return 100 - 100/(1+rs)

def bollinger_bands(df, period=20, std_dev=4):
    if len(df) < (period + 1):
        return []
    mid = df['c'].rolling(period).mean()
    std = df['c'].rolling(period).std()
    upper = mid + std_dev*std
    lower = mid - std_dev*std
    return upper, mid, lower

# ==== 4. SWING HIGH/LOW ====
def find_swings(df, lookback=2):
    if len(df) < (lookback*2 + 1):
        return []
    swings=[]
    for i in range(lookback, len(df)-lookback):
        high, low = df['h'][i], df['l'][i]
        if high==max(df['h'][i-lookback:i+lookback+1]):
            swings.append(('High', i, high))
        elif low==min(df['l'][i-lookback:i+lookback+1]):
            swings.append(('Low', i, low))
    highs=[s for s in swings if s[0]=='High'][-3:]
    lows=[s for s in swings if s[0]=='Low'][-3:]
    return highs+lows

# ==== 5. PRICE ACTION SIGNAL ====
def market_structure(df):
    swings=find_swings(df)
    structure="Sideway"
    if len(swings)>=4:
        highs=[s for s in swings if s[0]=='High'][-2:]
        lows=[s for s in swings if s[0]=='Low'][-2:]
        if highs and lows:
            if highs[-1][2]>highs[-2][2] and lows[-1][2]>lows[-2][2]:
                structure="Uptrend"
            elif highs[-1][2]<highs[-2][2] and lows[-1][2]<lows[-2][2]:
                structure="Downtrend"
    return structure

def liquidity_sweep(df):
    if len(df) < 3:
        return None  # chưa đủ dữ liệu
    for i in range(len(df)-3, len(df)-1):
        body = abs(df['c'][i]-df['o'][i])
        wick_high = df['h'][i]-max(df['c'][i], df['o'][i])
        wick_low = min(df['c'][i], df['o'][i])-df['l'][i]
        if wick_high >= 2*body:
            return "Sell-side liquidity sweep"
        if wick_low >= 2*body:
            return "Buy-side liquidity sweep"
    return None

def confirm_candle(df):
    if len(df) < 2:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    if last['c'] > last['o'] and last['c'] > (prev['o']+prev['c'])/2:
        return "Bullish confirm"
    elif last['c'] < last['o'] and last['c'] < (prev['o']+prev['c'])/2:
        return "Bearish confirm"
    return None

def find_order_blocks(df, atr_val):
    if len(df) < 20:
        return []
    ob_list=[]
    for i in range(2,len(df)-2):
        if df['c'][i]<df['o'][i] and df['c'][i+1]>df['h'][i]:
            ob_list.append({"type":"Bullish","low":df['l'][i],"high":df['h'][i],"atr":atr_val[i]})
        elif df['c'][i]>df['o'][i] and df['c'][i+1]<df['l'][i]:
            ob_list.append({"type":"Bearish","low":df['l'][i],"high":df['h'][i],"atr":atr_val[i]})
    return ob_list[-3:]

def trade_plan(signal_type, ob):
    atr_val=ob['atr']
    if signal_type=="BUY":
        entry=ob['low']
        sl=entry-1.5*atr_val
        tp=entry+3*atr_val
    else:
        entry=ob['high']
        sl=entry+1.5*atr_val
        tp=entry-3*atr_val
    rr=round(abs(tp-entry)/abs(entry-sl),2)
    return entry, sl, tp, rr

def pa_signal(df):
    atr_val=atr(df)
    structure=market_structure(df)
    sweep=liquidity_sweep(df)
    confirm=confirm_candle(df)
    ob_list=find_order_blocks(df, atr_val)
    signal, reason, plan = "WAIT", [], None
    if not ob_list: return signal, reason, structure, sweep, plan
    latest_ob=ob_list[-1]
    if structure=="Uptrend" and confirm=="Bullish confirm":
        if sweep=="Buy-side liquidity sweep" or latest_ob["type"]=="Bullish":
            signal="BUY"; reason.append("Uptrend + Demand + Sweep")
            entry, sl, tp, rr = trade_plan("BUY", latest_ob)
            plan={"entry":entry,"sl":sl,"tp":tp,"rr":rr}
    elif structure=="Downtrend" and confirm=="Bearish confirm":
        if sweep=="Sell-side liquidity sweep" or latest_ob["type"]=="Bearish":
            signal="SELL"; reason.append("Downtrend + Supply + Sweep")
            entry, sl, tp, rr = trade_plan("SELL", latest_ob)
            plan={"entry":entry,"sl":sl,"tp":tp,"rr":rr}
        if signal=="WAIT": reason.append("No valid PA setup")
    return signal, reason, structure, sweep, plan

# ==== 6. GENERATE REPORT MESSAGE ====
def generate_report(symbol, tf, df, signal, reason, plan):
    swings = find_swings(df)
    highs = [s for s in swings if s[0]=='High']
    lows = [s for s in swings if s[0]=='Low']

    price = df['c'].iloc[-1]
    atr_val = atr(df).iloc[-1]
    r = rsi(df).iloc[-1]
    ema34, ema89, ema200 = ema(df,34).iloc[-1], ema(df,89).iloc[-1], ema(df,200).iloc[-1]
    bb_upper, bb_mid, bb_lower = bollinger_bands(df)
    bb_upper, bb_mid, bb_lower = bb_upper.iloc[-1], bb_mid.iloc[-1], bb_lower.iloc[-1]

    highs_text = "\n".join([f"-Đỉnh tại {h[2]:.2f}, index {h[1]}" for h in highs])
    lows_text = "\n".join([f"-Đáy tại {l[2]:.2f}, index {l[1]}" for l in lows])
    plan_text = f"Entry: {plan['entry']:.2f}, SL: {plan['sl']:.2f}, TP: {plan['tp']:.2f}, RR: {plan['rr']}" if plan else "-"

    msg = f"""
📊 {symbol} ({tf})
Price: {price:.2f}

RSI(14): {r:.2f}
ATR(14): {atr_val:.2f}
EMA34: {ema34:.2f}, EMA89: {ema89:.2f}, EMA200: {ema200:.2f}
Bollinger Bands: Upper={bb_upper:.2f}, Mid={bb_mid:.2f}, Lower={bb_lower:.2f}

Signal: {signal}
Reason: {', '.join(reason) if reason else '-'}
Trade Plan: {plan_text}

3 Đỉnh gần nhất:
{highs_text}

3 Đáy gần nhất:
{lows_text}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    return msg

# ==== 7. PROCESS SYMBOL/TF ====
def process_symbol_tf(sym, tf):
    try:
        df = fetch_klines(sym, tf)
        if df.empty:
            print(f"⚠️ No data for {sym}-{tf}, skipping...")
            return  # bỏ qua nếu không có dữ liệu
        price = df['c'].iloc[-1]  # an toàn vì df đã kiểm tra
        sig, reason, structure, sweep, plan, next_swing = pa_signal(df)

        # ... tiếp tục logic
    except Exception as e:
        print(f"Error processing {sym}-{tf}: {e}")
    # Nếu PA signal hoặc giá chạm Bollinger, gửi ngay
    bb_upper, bb_mid, bb_lower = bollinger_bands(df)
    price = df['c'].iloc[-1]
    bollinger_alert = price >= bb_upper.iloc[-1] or price <= bb_lower.iloc[-1]

    if signal != "WAIT" or bollinger_alert:
        msg = generate_report(symbol, tf, df, signal, reason, plan)
        send_telegram(msg)

    return df

# ==== 8. SCHEDULED JOBS ====
def job_every_5min():
    for sym in SYMBOLS:
        for tf in TIMEFRAMES.keys():
            process_symbol_tf(sym, tf)

def job_hourly_h4():
    for sym in SYMBOLS:
        df = fetch_klines(sym,"4h")
        signal, reason, structure, sweep, plan = pa_signal(df)
        msg = generate_report(sym,"4h",df,signal,reason,plan)
        send_telegram(msg)

def job_daily_d1():
    now = datetime.now()
    if now.hour < 8:
        for sym in SYMBOLS:
            df = fetch_klines(sym,"1d")
            signal, reason, structure, sweep, plan = pa_signal(df)
            msg = generate_report(sym,"1d",df,signal,reason,plan)
            send_telegram(msg)

def job_weekly_w1():
    now = datetime.now()
    if now.weekday() == 0 and now.hour < 8:  # Monday trước 8h
        for sym in SYMBOLS:
            df = fetch_klines(sym,"1w")
            signal, reason, structure, sweep, plan = pa_signal(df)
            msg = generate_report(sym,"1w",df,signal,reason,plan)
            send_telegram(msg)

# ==== 9. MAIN LOOP ====
if __name__=="__main__":
    print(f"Bot started at {datetime.now()}")
    send_start_message()
# Schedule
    schedule.every(5).minutes.do(job_every_5min)       # Tính toán M15/H4/D1/W1 mỗi 5 phút
    schedule.every().hour.at(":00").do(job_hourly_h4) # Báo cáo H4 mỗi giờ
    schedule.every().day.at("07:50").do(job_daily_d1) # D1 trước 8h sáng
    schedule.every().monday.at("07:50").do(job_weekly_w1) # W1 thứ 2 trước 8h

    while True:
        schedule.run_pending()
        time.sleep(5)
