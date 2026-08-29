import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from google import genai
from datetime import datetime, timedelta

st.set_page_config(page_title="Pro Scanner & Backtester", layout="wide")
st.title("📈 Multi-Strategy Scanner & Backtesting Engine")

# --- SIDEBAR: CONFIGURATION ---
st.sidebar.header("Configuration")
market_universe = st.sidebar.selectbox(
    "Select Watchlist",
    ["Nifty 50 Sample", "Mid/Small Cap Sample", "Custom List"]
)

if market_universe == "Nifty 50 Sample":
    TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "SBIN.NS", "LT.NS"]
elif market_universe == "Mid/Small Cap Sample":
    TICKERS = ["KPITTECH.NS", "PERSISTENT.NS", "TRENT.NS", "DIXON.NS", "POLYCAB.NS", "SUZLON.NS", "KAYNES.NS", "CDSL.NS"]
else:
    custom_input = st.sidebar.text_area("Enter NSE Tickers (comma-separated)", "TATAMOTORS.NS, BEL.NS, HAL.NS")
    TICKERS = [x.strip() for x in custom_input.split(",") if x.strip()]

strategy_choice = st.sidebar.radio("Select Strategy", ["Scanner 1: 10/21 EMA Momentum", "Scanner 2: Techno-Funda 52W/ATH Breakout"])

# Optional Gemini Integration
st.sidebar.subheader("🤖 AI Trade Analyst (Optional)")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")

# --- CORE INDICATOR CALCULATIONS ---
def compute_technical_indicators(df):
    """Calculates all moving averages, RSI, and lookbacks."""
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Volume Averages
    df['Vol_SMA_21'] = df['Volume'].rolling(window=21).mean()
    df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Historical rolling highs (shifted by 1 to prevent lookahead bias)
    df['Max_High_200'] = df['High'].rolling(window=200).max().shift(1)
    df['Max_High_250'] = df['High'].rolling(window=250).max().shift(1)
    df['Max_High_2500'] = df['High'].rolling(window=min(len(df)-1, 2500)).max().shift(1)
    
    return df

# --- SCANNER 1: EMA CROSSOVER & MOMENTUM ---
def evaluate_scanner_1(df, market_cap):
    if len(df) < 220:
        return False, "Insufficient data (needs 200+ days)"
        
    c = df.iloc[-1]
    p = df.iloc[-2] # 1 day ago
    p20 = df.iloc[-21] if len(df) >= 21 else df.iloc[0] # 20 days ago
    
    cond1 = c['Close'] > c['EMA_10']
    cond2 = c['EMA_10'] > c['EMA_21']
    cond3 = p['EMA_10'] < p['EMA_21'] # Fresh Bullish Crossover
    cond4 = c['Volume'] > (c['Vol_SMA_21'] * 1.5)
    cond5 = c['Close'] > c['SMA_200']
    cond6 = c['Close'] > 30
    cond7 = c['Vol_SMA_21'] > 200000
    cond8 = c['Close'] < (p['Close'] * 1.06) # Not overextended (>6%)
    cond9 = (c['Close'] * c['Volume']) > 50000000 # Turnover > 5 Cr
    cond10 = c['SMA_50'] > c['SMA_200']
    cond11 = c['SMA_200'] > p20['SMA_200'] # 200 SMA sloping up
    cond12 = c['Close'] > (c['Max_High_200'] * 0.70)
    cond13 = c['RSI_14'] > 55
    cond14 = (market_cap < 250000000000) if market_cap else True # < ₹25,000 Cr
    
    passed = all([cond1, cond2, cond3, cond4, cond5, cond6, cond7, 
                  cond8, cond9, cond10, cond11, cond12, cond13, cond14])
    return passed, {
        "Close": round(c['Close'], 2),
        "RSI": round(c['RSI_14'], 1),
        "Vol/SMA21": round(c['Volume'] / c['Vol_SMA_21'], 2),
        "Turnover (Cr)": round((c['Close'] * c['Volume']) / 1e7, 2)
    }

# --- SCANNER 2: TECHNO-FUNDA BREAKOUT ---
def evaluate_scanner_2(df, info):
    if len(df) < 260:
        return False, "Insufficient data"
        
    c = df.iloc[-1]
    p = df.iloc[-2]
    
    # Fundamentals via yfinance info
    mcap_cr = (info.get('marketCap', 0) or 0) / 1e7
    promoter_pct = (info.get('heldPercentInsiders', 0) or 0) * 100
    de_ratio = (info.get('debtToEquity', 0) or 0) / 100
    roe = (info.get('returnOnEquity', 0) or 0) * 100
    profit_growth_yoy = (info.get('earningsQuarterlyGrowth', 0) or 0) * 100
    
    # ALL Conditions
    funda_ok = (
        mcap_cr > 1000 and
        c['Close'] > 50 and
        promoter_pct > 50 and
        de_ratio < 0.5 and
        roe > 15 and
        profit_growth_yoy > 15
    )
    volume_ok = c['Volume'] > (2 * p['Vol_SMA_20'])
    
    # ANY Condition: Within 3% of 250-day High OR ATH breakout
    near_52w_high = c['Close'] >= (0.97 * c['Max_High_250'])
    ath_breakout = c['Close'] >= c['Max_High_2500']
    price_ok = near_52w_high or ath_breakout
    
    passed = funda_ok and volume_ok and price_ok
    return passed, {
        "Close": round(c['Close'], 2),
        "Market Cap (Cr)": round(mcap_cr, 1),
        "Promoter %": round(promoter_pct, 1),
        "ROE %": round(roe, 1),
        "D/E": round(de_ratio, 2),
        "YoY Profit %": round(profit_growth_yoy, 1),
        "52W High Status": "ATH Breakout" if ath_breakout else "Near 52W High"
    }
    
def get_ai_analysis(ticker, metrics, api_key):
    """Passes the breakout data to Gemini for a fundamental/technical narrative."""
    if not api_key:
        return "No API key provided."
        
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an expert quantitative trader. The stock {ticker} just triggered a 
    bullish breakout on our scanner with the following metrics:
    {metrics}
    
    In 3 bullet points, provide a rapid risk-reward analysis of this setup. 
    Mention general sector headwinds or tailwinds if relevant.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI Analysis failed: {e}"
        
# --- UI ACTION BUTTONS ---
tab1, tab2 = st.tabs(["🔍 Live Scanner", "📊 Backtesting Sandbox"])

with tab1:
    st.subheader(f"Running: {strategy_choice}")
    if st.button("Run Market Scan"):
        results = []
        progress_bar = st.progress(0)
        
        for idx, ticker in enumerate(TICKERS):
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="3y")
                
                if not data.empty:
                    df = compute_technical_indicators(data)
                    info = stock.info
                    mcap = info.get('marketCap', 0)
                    
                    if "Scanner 1" in strategy_choice:
                        passed, metrics = evaluate_scanner_1(df, mcap)
                    else:
                        passed, metrics = evaluate_scanner_2(df, info)
                        
                    if passed:
                        metrics["Ticker"] = ticker
                        results.append(metrics)
            except Exception as e:
                pass
            progress_bar.progress((idx + 1) / len(TICKERS))
            
        if results:
            res_df = pd.DataFrame(results)
            st.success(f"Found {len(results)} matching opportunities!")
            st.dataframe(res_df.set_index("Ticker"), use_container_width=True)
        else:
            st.warning("No stocks passed all scanner filters today.")

with tab2:
    st.subheader("Historical Backtest Engine")
    test_ticker = st.selectbox("Select Ticker to Backtest", TICKERS)
    test_years = st.slider("Lookback Period (Years)", 1, 5, 3)
    
    if st.button("Run Backtest"):
        stock = yf.Ticker(test_ticker)
        df = stock.history(period=f"{test_years}y")
        
        if len(df) > 200:
            df = compute_technical_indicators(df)
            
            # Backtest Strategy 1 Logic:
            # Buy on fresh EMA 10 > EMA 21 with Close > SMA 200 & RSI > 55
            # Exit when Close drops below EMA 21
            trades = []
            in_pos = False
            entry_price, entry_date = 0, None
            
            for i in range(1, len(df)):
                curr = df.iloc[i]
                prev = df.iloc[i-1]
                
                # Signal Generation
                buy_signal = (
                    curr['EMA_10'] > curr['EMA_21'] and
                    prev['EMA_10'] <= prev['EMA_21'] and
                    curr['Close'] > curr['SMA_200'] and
                    curr['RSI_14'] > 55
                )
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
                        "Entry Price": entry_price,
                        "Exit Price": curr['Close'],
                        "Return %": pnl
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
                
                # Plot Cumulative PnL Curve
                tdf['Cumulative'] = (1 + tdf['Return %']/100).cumprod()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=tdf['Exit Date'], y=tdf['Cumulative'], mode='lines+markers', name='Equity Curve'))
                fig.update_layout(title="Strategy Equity Multiplier", xaxis_title="Date", yaxis_title="Portfolio Growth (1.0 = Base)")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(tdf)
            else:
                st.info("No completed trades triggered for the selected parameters.")
