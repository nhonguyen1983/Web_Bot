import requests, pandas as pd, numpy as np, time, os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
TIMEFRAMES = {"15m": 15, "4h": 240, "1d": 1440}
LIMIT = 250

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
def find_swings1(df, lookback=3):
    swings = []
    for i in range(lookback, len(df)-lookback):
        high, low = df['h'][i], df['l'][i]
        if high == max(df['h'][i-lookback:i+lookback+1]):
            swings.append(('High', i, high))
        elif low == min(df['l'][i-lookback:i+lookback+1]):
            swings.append(('Low', i, low))
    return swings[-6:]
def find_swings(df, lookback=2):
    """
    Xác định swing high / swing low dựa trên 5 nến (2 trái, 1 trung tâm, 2 phải)
    """
    swings = []
    for i in range(lookback, len(df)-lookback):
        high = df['h'][i]
        low = df['l'][i]

        # Swing High: cao nhất trong 5 nến
        if high == max(df['h'][i-lookback:i+lookback+1]):
            swings.append(('High', i, high))

        # Swing Low: thấp nhất trong 5 nến
        elif low == min(df['l'][i-lookback:i+lookback+1]):
            swings.append(('Low', i, low))

    # Lấy 3 đỉnh và 3 đáy mới nhất
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
    # khoảng cách nến giữa các swing
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
def send_telegram(symbol, tf, signal, reason, structure, sweep, plan, next_swing):
    msg=f"""
📊 {symbol} ({tf})
Trend: {structure}
Liquidity: {sweep}
Signal: {signal} 🚦
Reason: {', '.join(reason)}

🎯 ENTRY PLAN
Entry: {round(plan['entry'],2) if plan else '-'}
SL: {round(plan['sl'],2) if plan else '-'}
TP: {round(plan['tp'],2) if plan else '-'}
RR: {plan['rr'] if plan else '-'}

📈 NEXT SWING ESTIMATE
Next swing in ~{next_swing} candles

Time: {datetime.now().strftime('%H:%M:%S')}
"""
    requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                 params={"chat_id":CHAT_ID,"text":msg})

# ==== 12. MAIN LOOP ====
def analyze_all():
    for sym in SYMBOLS:
        for tf in TIMEFRAMES.keys():
            try:
                df = fetch_klines(sym, tf)
                sig, reason, structure, sweep, plan, next_swing = pa_signal(df)
                if sig!="WAIT":
                    send_telegram(sym, tf, sig, reason, structure, sweep, plan, next_swing)
                time.sleep(0.2)
            except Exception as e:
                print(f"Error {sym}-{tf}: {e}")

if __name__=="__main__":
    while True:
        print(f"⏱️ Checking signals at {datetime.now()}...")
        analyze_all()
        time.sleep(60)