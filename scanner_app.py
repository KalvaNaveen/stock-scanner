import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Custom CSV Scanner & Backtester", layout="wide")
st.title("📈 Custom List Scanner & Backtesting Engine")

# --- SIDEBAR: DATA SOURCE ---
st.sidebar.header("1. Upload Watchlist")
st.sidebar.write("Upload a CSV file containing your stock symbols.")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

TICKERS = []
if uploaded_file is not None:
    try:
        df_symbols = pd.read_csv(uploaded_file)
        
        # Smart column detection: looks for 'symbol' or 'ticker', otherwise defaults to the first column
        col_name = None
        for col in df_symbols.columns:
            if col.strip().lower() in ['symbol', 'ticker', 'symbols', 'tickers']:
                col_name = col
                break
        if not col_name:
            col_name = df_symbols.columns[0]
        
        raw_tickers = df_symbols[col_name].dropna().astype(str).tolist()
        
        # Ensure .NS suffix for Indian Stocks (yfinance format)
        TICKERS = [t.strip() + '.NS' if not t.strip().endswith('.NS') else t.strip() for t in raw_tickers]
        st.sidebar.success(f"Successfully loaded {len(TICKERS)} tickers from '{col_name}' column.")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
else:
    st.sidebar.info("Awaiting CSV file upload.")

st.sidebar.header("2. Strategy Configuration")
strategy_choice = st.sidebar.radio("Select Strategy", [
    "Scanner 1: 10/21 EMA Momentum", 
    "Scanner 2: Techno-Funda Breakout"
])

# --- CORE INDICATOR CALCULATIONS ---
@st.cache_data(show_spinner=False)
def compute_technical_indicators(df):
    if len(df) == 0:
        return df
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    df['Vol_SMA_21'] = df['Volume'].rolling(window=21).mean()
    df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    df['Max_High_200'] = df['High'].rolling(window=200).max().shift(1)
    df['Max_High_250'] = df['High'].rolling(window=250).max().shift(1)
    df['Max_High_2500'] = df['High'].rolling(window=min(len(df)-1, 2500)).max().shift(1)
    return df

# --- SCANNER LOGIC ---
def evaluate_scanner_1(df, market_cap):
    if len(df) < 220: return False, {}
    c, p, p20 = df.iloc[-1], df.iloc[-2], df.iloc[-21]
    
    conds = [
        c['Close'] > c['EMA_10'],
        c['EMA_10'] > c['EMA_21'],
        p['EMA_10'] < p['EMA_21'],
        c['Volume'] > (c['Vol_SMA_21'] * 1.5),
        c['Close'] > c['SMA_200'],
        c['Close'] > 30,
        c['Vol_SMA_21'] > 200000,
        c['Close'] < (p['Close'] * 1.06),
        (c['Close'] * c['Volume']) > 50000000,
        c['SMA_50'] > c['SMA_200'],
        c['SMA_200'] > p20['SMA_200'],
        c['Close'] > (c['Max_High_200'] * 0.70),
        c['RSI_14'] > 55,
        (market_cap < 250000000000) if market_cap else True
    ]
    if all(conds):
        return True, {
            "Close": round(c['Close'], 2), 
            "RSI": round(c['RSI_14'], 1), 
            "Vol/SMA21": round(c['Volume'] / c['Vol_SMA_21'], 2),
            "Turnover (Cr)": round((c['Close'] * c['Volume']) / 1e7, 2)
        }
    return False, {}

def evaluate_scanner_2(df, info):
    if len(df) < 260: return False, {}
    c, p = df.iloc[-1], df.iloc[-2]
    
    mcap_cr = (info.get('marketCap', 0) or 0) / 1e7
    promoter_pct = (info.get('heldPercentInsiders', 0) or 0) * 100
    de_ratio = (info.get('debtToEquity', 0) or 0) / 100
    roe = (info.get('returnOnEquity', 0) or 0) * 100
    profit_growth_yoy = (info.get('earningsQuarterlyGrowth', 0) or 0) * 100
    
    funda_ok = (mcap_cr > 1000 and c['Close'] > 50 and promoter_pct > 50 and de_ratio < 0.5 and roe > 15 and profit_growth_yoy > 15)
    volume_ok = c['Volume'] > (2 * p['Vol_SMA_20'])
    price_ok = (c['Close'] >= (0.97 * c['Max_High_250'])) or (c['Close'] >= c['Max_High_2500'])
    
    if funda_ok and volume_ok and price_ok:
        return True, {
            "Close": round(c['Close'], 2), 
            "Market Cap (Cr)": round(mcap_cr, 1), 
            "ROE %": round(roe, 1),
            "YoY Profit %": round(profit_growth_yoy, 1)
        }
    return False, {}

# --- UI TABS ---
if not TICKERS:
    st.warning("👈 Please upload a CSV file containing your stock symbols to start scanning and backtesting.")
    st.stop()

tab1, tab2 = st.tabs(["🔍 Live Scanner", "📊 Backtesting Sandbox"])

with tab1:
    st.subheader(f"Running: {strategy_choice}")
    if st.button("Run Market Scan on Uploaded List"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, ticker in enumerate(TICKERS):
            status_text.text(f"Scanning {ticker}...")
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="3y")
                if not data.empty:
                    df = compute_technical_indicators(data)
                    passed = False
                    
                    if "Scanner 1" in strategy_choice:
                        passed, metrics = evaluate_scanner_1(df, stock.info.get('marketCap', 0))
                    else:
                        passed, metrics = evaluate_scanner_2(df, stock.info)
                    
                    if passed:
                        metrics["Ticker"] = ticker
                        results.append(metrics)
            except Exception:
                pass
            
            progress_bar.progress((idx + 1) / len(TICKERS))
        
        status_text.text("Scan Complete!")
        if results:
            st.success(f"Found {len(results)} matching opportunities from your list!")
            st.dataframe(pd.DataFrame(results).set_index("Ticker"), use_container_width=True)
        else:
            st.warning("No stocks from your uploaded list passed the filters today.")

with tab2:
    st.subheader("Historical Backtest Engine (Filtered by Uploaded List)")
    test_ticker = st.selectbox("Select a Ticker to Backtest", TICKERS)
    test_years = st.slider("Lookback Period (Years)", 1, 5, 3)
    
    if st.button("Run Backtest on Selected Ticker"):
        with st.spinner(f"Fetching data and backtesting {test_ticker}..."):
            stock = yf.Ticker(test_ticker)
            df = stock.history(period=f"{test_years}y")
            
            if len(df) > 200:
                df = compute_technical_indicators(df)
                trades = []
                in_pos = False
                entry_price, entry_date = 0, None
                
                # Simple momentum backtest logic
                for i in range(1, len(df)):
                    curr = df.iloc[i]
                    prev = df.iloc[i-1]
                    
                    buy_signal = (curr['EMA_10'] > curr['EMA_21'] and 
                                  prev['EMA_10'] <= prev['EMA_21'] and 
                                  curr['Close'] > curr['SMA_200'] and 
                                  curr['RSI_14'] > 55)
                    sell_signal = curr['Close'] < curr['EMA_21']
                    
                    if buy_signal and not in_pos:
                        in_pos = True
                        entry_price = curr['Close']
                        entry_date = df.index[i]
                    elif sell_signal and in_pos:
                        in_pos = False
                        pnl = ((curr['Close'] - entry_price) / entry_price) * 100
                        trades.append({
                            "Entry Date": entry_date, 
                            "Exit Date": df.index[i], 
                            "Entry Price": round(entry_price, 2), 
                            "Exit Price": round(curr['Close'], 2), 
                            "Return %": round(pnl, 2)
                        })
                        
                if trades:
                    tdf = pd.DataFrame(trades)
                    wins = tdf[tdf['Return %'] > 0]
                    win_rate = (len(wins) / len(tdf)) * 100
                    total_return = tdf['Return %'].sum()
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Trades", len(tdf))
                    c2.metric("Win Rate", f"{win_rate:.1f}%")
                    c3.metric("Cumulative Return", f"{total_return:.2f}%")
                    
                    tdf['Cumulative Growth'] = (1 + tdf['Return %']/100).cumprod()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=tdf['Exit Date'], y=tdf['Cumulative Growth'], mode='lines+markers', name='Equity Curve'))
                    fig.update_layout(title="Strategy Equity Multiplier", xaxis_title="Date", yaxis_title="Portfolio Growth (1.0 = Base)")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(tdf)
                else:
                    st.info("No completed trades triggered for the selected parameters.")
            else:
                st.error("Insufficient historical data for this ticker to run a meaningful backtest.")
