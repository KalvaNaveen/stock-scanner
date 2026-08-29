import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from google import genai
import datetime

st.set_page_config(page_title="Pro Cloud Scanner & AI Analyst", layout="wide")
st.title("☁️ Cloud Stock Scanner & AI Backtesting Engine")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("1. Upload Watchlist")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

TICKERS = []
if uploaded_file is not None:
    try:
        df_symbols = pd.read_csv(uploaded_file)
        col_name = next((col for col in df_symbols.columns if col.strip().lower() in ['symbol', 'ticker', 'symbols', 'tickers']), df_symbols.columns[0])
        raw_tickers = df_symbols[col_name].dropna().astype(str).tolist()
        TICKERS = [t.strip() + '.NS' if not t.strip().endswith('.NS') else t.strip() for t in raw_tickers]
        st.sidebar.success(f"Loaded {len(TICKERS)} stocks.")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")

st.sidebar.header("2. Strategy Configuration")
strategy_choice = st.sidebar.radio("Select Strategy", ["Scanner 1: 10/21 EMA Momentum", "Scanner 2: Techno-Funda Breakout"])

# --- NEW: TIME MACHINE (HISTORICAL SCAN) ---
st.sidebar.header("3. Time Machine (Historical Scan)")
today = datetime.date.today()
scan_date = st.sidebar.date_input("Run scan as of date:", value=today, max_value=today)

st.sidebar.header("4. AI Integration (Optional)")
gemini_api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

# --- AI ANALYST FUNCTION ---
def get_ai_analysis(ticker, metrics, api_key):
    if not api_key: return None
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
        You are an expert quantitative trader. The Indian stock {ticker} triggered a bullish breakout on our scanner with these metrics: {metrics}. 
        In exactly 3 bullet points, provide a rapid risk-reward analysis of this setup.
        """
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e:
        return f"AI Analysis failed: {e}"

# --- CORE INDICATOR CALCULATIONS ---
@st.cache_data(show_spinner=False)
def compute_technical_indicators(df):
    if len(df) == 0: return df
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
        c['Close'] > c['EMA_10'], c['EMA_10'] > c['EMA_21'], p['EMA_10'] < p['EMA_21'],
        c['Volume'] > (c['Vol_SMA_21'] * 1.5), c['Close'] > c['SMA_200'], c['Close'] > 30,
        c['Vol_SMA_21'] > 200000, c['Close'] < (p['Close'] * 1.06), (c['Close'] * c['Volume']) > 50000000,
        c['SMA_50'] > c['SMA_200'], c['SMA_200'] > p20['SMA_200'], c['Close'] > (c['Max_High_200'] * 0.70),
        c['RSI_14'] > 55, (market_cap < 250000000000) if market_cap else True
    ]
    if all(conds):
        return True, {"Close": round(c['Close'], 2), "RSI": round(c['RSI_14'], 1), "Turnover (Cr)": round((c['Close'] * c['Volume']) / 1e7, 2)}
    return False, {}

def evaluate_scanner_2(df, info):
    if len(df) < 260: return False, {}
    c, p = df.iloc[-1], df.iloc[-2]
    mcap_cr = (info.get('marketCap', 0) or 0) / 1e7
    funda_ok = (mcap_cr > 1000 and c['Close'] > 50 and (info.get('heldPercentInsiders', 0) or 0)*100 > 50 and (info.get('returnOnEquity', 0) or 0)*100 > 15)
    volume_ok = c['Volume'] > (2 * p['Vol_SMA_20'])
    price_ok = (c['Close'] >= (0.97 * c['Max_High_250'])) or (c['Close'] >= c['Max_High_2500'])
    if funda_ok and volume_ok and price_ok:
        return True, {"Close": round(c['Close'], 2), "Market Cap (Cr)": round(mcap_cr, 1)}
    return False, {}

# --- UI TABS ---
if not TICKERS:
    st.warning("👈 Please upload your CSV file containing your stock symbols to begin.")
    st.stop()

tab1, tab2 = st.tabs(["🔍 Live/Historical Scanner", "📊 Backtesting Sandbox"])

with tab1:
    st.subheader(f"Running: {strategy_choice} | As of: {scan_date.strftime('%Y-%m-%d')}")
    if st.button("Start Cloud Scan"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        target_date = pd.to_datetime(scan_date)
        
        for idx, ticker in enumerate(TICKERS):
            status_text.text(f"Scanning {ticker}...")
            try:
                stock = yf.Ticker(ticker)
                # Fetch 5 years of data so historical scans have enough lookback period
                data = stock.history(period="5y") 
                
                if not data.empty:
                    # Strip timezones for clean date slicing
                    data.index = data.index.tz_localize(None)
                    df = compute_technical_indicators(data)
                    
                    # TIME MACHINE: Truncate dataframe up to the user's selected date
                    df_historical = df.loc[:target_date]
                    
                    if not df_historical.empty and len(df_historical) >= 260:
                        passed = False
                        if "Scanner 1" in strategy_choice:
                            passed, metrics = evaluate_scanner_1(df_historical, stock.info.get('marketCap', 0))
                        else:
                            passed, metrics = evaluate_scanner_2(df_historical, stock.info)
                        
                        if passed:
                            metrics["Ticker"] = ticker
                            # Show the actual date it triggered (helpful if they pick a weekend)
                            metrics["Trigger Date"] = df_historical.index[-1].strftime('%Y-%m-%d')
                            results.append(metrics)
            except Exception:
                pass
            progress_bar.progress((idx + 1) / len(TICKERS))
        
        status_text.text("Scan Complete!")
        if results:
            st.success(f"Found {len(results)} matching breakouts as of {scan_date.strftime('%Y-%m-%d')}!")
            st.dataframe(pd.DataFrame(results).set_index("Ticker"), use_container_width=True)
            
            if gemini_api_key:
                st.subheader("🤖 Gemini Trade Analyst Insights")
                for res in results[:3]:
                    with st.expander(f"AI Analysis for {res['Ticker']}", expanded=True):
                        with st.spinner("Gemini is analyzing the chart..."):
                            st.write(get_ai_analysis(res['Ticker'], res, gemini_api_key))
        else:
            st.warning(f"No stocks passed all strict filters on {scan_date.strftime('%Y-%m-%d')}.")

with tab2:
    st.subheader("Historical Backtest Engine")
    test_ticker = st.selectbox("Select a Ticker to Backtest", TICKERS)
    
    if st.button("Run Backtest"):
        with st.spinner(f"Backtesting {test_ticker}..."):
            stock = yf.Ticker(test_ticker)
            df = stock.history(period="5y")
            
            if len(df) > 200:
                df = compute_technical_indicators(df)
                trades = []
                in_pos, entry_price, entry_date = False, 0, None
                
                for i in range(1, len(df)):
                    curr, prev = df.iloc[i], df.iloc[i-1]
                    buy_signal = (curr['EMA_10'] > curr['EMA_21'] and prev['EMA_10'] <= prev['EMA_21'] and curr['Close'] > curr['SMA_200'])
                    sell_signal = curr['Close'] < curr['EMA_21']
                    
                    if buy_signal and not in_pos:
                        in_pos, entry_price, entry_date = True, curr['Close'], df.index[i]
                    elif sell_signal and in_pos:
                        in_pos = False
                        pnl = ((curr['Close'] - entry_price) / entry_price) * 100
                        trades.append({"Entry": entry_date.strftime('%Y-%m-%d'), "Exit": df.index[i].strftime('%Y-%m-%d'), "Return %": round(pnl, 2)})
                        
                if trades:
                    tdf = pd.DataFrame(trades)
                    wins = tdf[tdf['Return %'] > 0]
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Trades", len(tdf))
                    c2.metric("Win Rate", f"{(len(wins)/len(tdf))*100:.1f}%")
                    c3.metric("Cumulative Return", f"{tdf['Return %'].sum():.2f}%")
                    
                    tdf['Equity'] = (1 + tdf['Return %']/100).cumprod()
                    fig = go.Figure(go.Scatter(x=tdf['Exit'], y=tdf['Equity'], mode='lines+markers', name='Equity Curve'))
                    fig.update_layout(title="Strategy Portfolio Growth (1.0 = Base)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No trades triggered.")
            else:
                st.error("Not enough historical data.")
