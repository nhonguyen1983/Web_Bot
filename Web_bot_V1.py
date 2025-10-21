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
TELEGRAM_ALLOWED_TFS = ['1d', '1w']  # chỉ gửi trade signal D1/W1

# === SYMBOLS & TIMEFRAMES ===
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
TIMEFRAMES = {"15m": 15, "4h": 240, "1d": 1440, "1w": 10080}
LIMIT = 250
def send_start_message():
    msg = f"🤖 Bot started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nHi! Bot đang chạy và sẽ gửi báo cáo/trade signal theo lịch."
    send_telegram_message(msg, timeframe='start')
# ==== 1. Fetch Candle Data ====
def fetch_klines(symbol, tf):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={tf}&limit={LIMIT}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['t','o','h','l','c','v','ct','q','n','tb','tq','ig'])
    df = df[['t','o','h','l','c','v']].astype(float)
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    return df

# ==== 2. ATR ====
def atr(df, period=14):
    high_low = df['h'] - df['l']
    high_close = np.abs(df['h'] - df['c'].shift())
    low_close = np.abs(df['l'] - df['c'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ==== 3. Swing High / Low ====
def find_swings(df, lookback=2):
    swings = []
    for i in range(lookback, len(df)-lookback):
        high = df['h'][i]
        low = df['l'][i]
        if high == max(df['h'][i-lookback:i+lookback+1]):
            swings.append(('High', i, high))
        elif low == min(df['l'][i-lookback:i+lookback+1]):
            swings.append(('Low', i, low))
    highs = [s for s in swings if s[0] == 'High'][-3:]
    lows = [s for s in swings if s[0] == 'Low'][-3:]
    return highs + lows

# ==== 4. Market Structure ====
def market_structure(df):
    swings = find_swings(df)
    structure = "Sideway"
    if len(swings) >= 4:
        highs = [s for s in swings if s[0]=='High'][-2:]
        lows = [s for s in swings if s[0]=='Low'][-2:]
        if highs and lows:
            if highs[-1][2]>highs[-2][2] and lows[-1][2]>lows[-2][2]:
                structure="Uptrend"
            elif highs[-1][2]<highs[-2][2] and lows[-1][2]<lows[-2][2]:
                structure="Downtrend"
    return structure

# ==== 5. Liquidity Sweep ====
def liquidity_sweep(df):
    for i in range(len(df)-3, len(df)-1):
        body = abs(df['c'][i]-df['o'][i])
        wick_high = df['h'][i]-max(df['c'][i], df['o'][i])
        wick_low = min(df['c'][i], df['o'][i])-df['l'][i]
        if wick_high>=2*body: return "Sell-side liquidity sweep"
        if wick_low>=2*body: return "Buy-side liquidity sweep"
    return None

# ==== 6. Order Block ====
def find_order_blocks(df, atr_val):
    ob_list=[]
    for i in range(2, len(df)-2):
        if df['c'][i]<df['o'][i] and df['c'][i+1]>df['h'][i]:
            ob_list.append({"type":"Bullish","low":df['l'][i],"high":df['h'][i],"atr":atr_val[i]})
        elif df['c'][i]>df['o'][i] and df['c'][i+1]<df['l'][i]:
            ob_list.append({"type":"Bearish","low":df['l'][i],"high":df['h'][i],"atr":atr_val[i]})
    return ob_list[-3:]

# ==== 7. Confirm Candle ====
def confirm_candle(df):
    last, prev = df.iloc[-1], df.iloc[-2]
    if last['c']>last['o'] and last['c']>(prev['o']+prev['c'])/2:
        return "Bullish confirm"
    elif last['c']<last['o'] and last['c']<(prev['o']+prev['c'])/2:
        return "Bearish confirm"
    return None

# ==== 8. Trade Plan ====
def trade_plan(signal_type, ob):
    atr_val = ob['atr']
    if signal_type=="BUY":
        entry = ob['low']
        sl = entry - 1.5*atr_val
        tp = entry + 3*atr_val
    else:
        entry = ob['high']
        sl = entry + 1.5*atr_val
        tp = entry - 3*atr_val
    rr = round(abs(tp-entry)/abs(entry-sl),2)
    return entry, sl, tp, rr

# ==== 9. Predict next swing ====
def predict_next_swing(df, atr_val):
    swings = find_swings(df)
    if len(swings)<3: return None
    distances = [swings[i][1]-swings[i-1][1] for i in range(1,len(swings))]
    mean_dist = np.mean(distances)
    last_swing = swings[-1][1]
    candles_since = len(df)-last_swing
    atr_recent = atr_val.iloc[-1]
    atr_mean = atr_val.mean()
    adj_factor = atr_mean/atr_recent if atr_recent!=0 else 1
    est_remaining = max(int((mean_dist - candles_since)*adj_factor),0)
    return est_remaining

# ==== 10. Main Price Action ====
def pa_signal(df):
    atr_val = atr(df)
    structure = market_structure(df)
    sweep = liquidity_sweep(df)
    confirm = confirm_candle(df)
    ob_list = find_order_blocks(df, atr_val)
    next_swing = predict_next_swing(df, atr_val)

    signal, reason, plan = "WAIT", [], None
    if not ob_list: return signal, reason, structure, sweep, plan, next_swing

    latest_ob = ob_list[-1]

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
    return signal, reason, structure, sweep, plan, next_swing

# ==== 11. Telegram Send ====
def send_telegram_message(message: str, timeframe: str = ''):
    if not TELEGRAM_ENABLED: return False
    tf = timeframe.lower()
    if TELEGRAM_BOT_TOKEN is None or TELEGRAM_CHAT_ID is None:
        print("Telegram token/chat not configured")
    return False
    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
        resp = requests.post(url, data=payload, timeout=10)
        return resp.ok
    except Exception as e:
        print("send_telegram_message error:", e)
        return False

def generate_report_message(symbol, tf, sig, reason, structure, sweep, plan, next_swing):
    msg = f"""
📊 {symbol} ({tf})
Trend: {structure}
Liquidity: {sweep}
Signal: {sig} 🚦
Reason: {', '.join(reason)}

🎯 ENTRY PLAN
Entry: {round(plan['entry'],2) if plan else '-'}
SL: {round(plan['sl'],2) if plan else '-'}
TP: {round(plan['tp'],2) if plan else '-'}
RR: {plan['rr'] if plan else '-'}

📈 NEXT SWING ESTIMATE
Next swing in ~{next_swing} candles

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    return msg

def process_symbol_tf(sym, tf):
    try:
        df = fetch_klines(sym, tf)
        sig, reason, structure, sweep, plan, next_swing = pa_signal(df)

        # Gửi báo cáo 4h
        if tf=="4h":
            msg = generate_report_message(sym, tf, sig, reason, structure, sweep, plan, next_swing)
            send_telegram_message(msg, timeframe=tf)
            print(f"🕒 4h report sent for {sym}-{tf}")

        # Gửi trade signal D1/W1
        if tf in TELEGRAM_ALLOWED_TFS and sig!="WAIT":
            msg = generate_report_message(sym, tf, sig, reason, structure, sweep, plan, next_swing)
            send_telegram_message(msg, timeframe=tf)
            print(f"🚦 Signal sent for {sym}-{tf}: {sig}")

    except Exception as e:
        print(f"Error processing {sym}-{tf}: {e}")

def analyze_all_symbols():
    for sym in SYMBOLS:
        for tf in TIMEFRAMES.keys():
            process_symbol_tf(sym, tf)

# ==== MAIN SCHEDULE LOOP ====
schedule.every().hour.at(":00").do(analyze_all_symbols)

if __name__=="__main__":
    print(f"Bot started at {datetime.now()}")
    send_start_message()  # gửi lời chào Telegram ngay khi start
    while True:
        schedule.run_pending()
        time.sleep(5)
