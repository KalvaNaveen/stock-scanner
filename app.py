import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import re
from google import genai
import datetime
import os

# ==========================================
# 1. UI OVERHAUL & PERSISTENT STATE
# ==========================================
st.set_page_config(page_title="Quant Autonomous Engine", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0B0E14; color: #E2E8F0; }
    h1, h2, h3 { color: #FFFFFF; font-family: 'Inter', sans-serif; }
    .stButton>button { background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%); color: #000; font-weight: bold; border: none; }
    div[data-testid="stMetricValue"] { color: #00FF88 !important; font-size: 2rem !important; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Autonomous Quant & Paper Trading Engine")

# --- AUTO-SAVE INITIALIZATION ---
SCAN_FILE = "latest_scan.csv"
PORTFOLIO_FILE = "portfolio_backup.csv"

if 'chartink_data' not in st.session_state:
    if os.path.exists(SCAN_FILE):
        st.session_state['chartink_data'] = pd.read_csv(SCAN_FILE)
    else:
        st.session_state['chartink_data'] = pd.DataFrame()

if 'portfolio' not in st.session_state:
    if os.path.exists(PORTFOLIO_FILE):
        st.session_state['portfolio'] = pd.read_csv(PORTFOLIO_FILE)
    else:
        st.session_state['portfolio'] = pd.DataFrame(columns=['Symbol', 'Status', 'Entry Date', 'Entry Price', 'Shares', 'Max Reached', 'Trailing SL', 'Current LTP', 'Unrealized P&L'])

# ==========================================
# 2. CORE FUNCTIONS
# ==========================================
def get_ai_analysis(ticker, company, api_key):
    if not api_key: return "⚠️ Please enter your Gemini API key in the sidebar."
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"Analyze {ticker} ({company}) for a momentum breakout. Give 3 short bullet points on risk-reward and sector trends."
        return client.models.generate_content(model='gemini-2.5-flash', contents=prompt).text
    except Exception as e:
        return f"AI Analysis failed: {e}"

@st.cache_data(ttl=300, show_spinner=False)
def scrape_chartink(screener_url, manual_clause=""):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        with requests.Session() as s:
            r = s.get(screener_url, headers=headers)
            soup = BeautifulSoup(r.text, 'html.parser')
            csrf = soup.find('meta', {'name': 'csrf-token'})['content'] if soup.find('meta', {'name': 'csrf-token'}) else ""
            
            scan_clause = manual_clause
            if not scan_clause:
                m = re.search(r'scan_clause\s*:\s*\'(.*?)\'', r.text) or re.search(r'name="scan_clause"\s+value="(.*?)"', r.text)
                scan_clause = m.group(1) if m else ""

            if not scan_clause: return pd.DataFrame(), "Could not auto-detect scan clause."

            res = s.post('https://chartink.com/screener/process', 
                         data={'scan_clause': scan_clause}, 
                         headers={'x-csrf-token': csrf, 'X-Requested-With': 'XMLHttpRequest', **headers})
            
            if res.status_code == 200:
                data = res.json()
                if 'error' in data: return pd.DataFrame(), f"Chartink Error: {data['error']}"
                if 'data' in data:
                    if len(data['data']) == 0: return pd.DataFrame(), "Success, but 0 stocks matched today."
                    df = pd.DataFrame(data['data'])
                    df['nsecode'] = df['nsecode'] + '.NS'
                    return df, "Success"
            return pd.DataFrame(), f"Rejected: {res.status_code}"
    except Exception as e:
        return pd.DataFrame(), str(e)

def compute_historical_indicators(df):
    if len(df) < 200: return df
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Avg_Vol_20'] = df['Volume'].rolling(window=20).mean()
    df['RVOL'] = df['Volume'] / df['Avg_Vol_20'].replace(0, np.nan)
    return df

def run_trailing_backtest(df_price, capital, sl_pct):
    trades = []
    in_pos, entry_price, entry_date, shares, highest_price, current_sl = False, 0, None, 0, 0, 0
    for i in range(1, len(df_price)):
        curr, prev = df_price.iloc[i], df_price.iloc[i-1]
        if not in_pos and curr['EMA_10'] > curr['EMA_20'] and prev['EMA_10'] <= prev['EMA_20'] and curr['Close'] > curr['SMA_200']:
            in_pos, entry_price, entry_date = True, curr['Close'], df_price.index[i]
            shares = capital // entry_price
            highest_price, current_sl = entry_price, entry_price * (1 - sl_pct)
        elif in_pos:
            if curr['High'] > highest_price:
                highest_price = curr['High']
                new_sl = highest_price * (1 - sl_pct)
                if new_sl > current_sl: current_sl = new_sl 
            if curr['Low'] <= current_sl:
                in_pos = False
                trades.append({
                    "Entry Date": entry_date.strftime('%Y-%m-%d'), "Exit Date": df_price.index[i].strftime('%Y-%m-%d'),
                    "Entry Price": round(entry_price, 2), "Exit Price": round(current_sl, 2), "Shares": int(shares), 
                    "P&L": round((current_sl - entry_price) * shares, 2)
                })
    return trades

# ==========================================
# 3. SIDEBAR CONFIG & SECRETS
# ==========================================
with st.sidebar:
    st.header("⚙️ Core Settings")
    capital = st.number_input("Capital Per Trade (₹)", value=100000, step=10000)
    sl_percent = st.number_input("Trailing Risk (%)", value=2.0, step=0.5) / 100
    
    secret_api = st.secrets.get("GEMINI_API_KEY", "")
    if secret_api:
        gemini_key = secret_api
        st.success("✅ Gemini Key loaded.")
    else:
        gemini_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    st.header("🔍 Chartink Scanner")
    secret_url = st.secrets.get("CHARTINK_URL", "https://chartink.com/screener/momentum-stocks-29112990")
    secret_clause = st.secrets.get("CHARTINK_CLAUSE", "")
    url_input = st.text_input("URL", value=secret_url)
    
    if secret_clause:
        manual_clause = secret_clause
        st.success("✅ Scan Clause loaded.")
    else:
        manual_clause = st.text_area("Manual Scan Clause (Bypass)")
        
    if st.button("🚀 RUN DAILY SCAN"):
        with st.spinner("Scraping live data..."):
            df, status = scrape_chartink(url_input, manual_clause)
            if not df.empty:
                st.session_state['chartink_data'] = df
                df.to_csv(SCAN_FILE, index=False) # <--- AUTO-SAVE SCAN TO DISK
                st.success(f"Scraped {len(df)} stocks!")
            else:
                st.error(status)
                
    st.markdown("---")
    st.header("⏱️ Historical Universe")
    st.write("Upload your CSV watchlist to run the Time Machine Backtester.")
    uploaded_universe = st.file_uploader("Upload CSV", type=['csv'])

# ==========================================
# 4. TABBED INTERFACE
# ==========================================
tab1, tab2, tab3 = st.tabs(["⚡ 1. Today's Scan & AI", "📓 2. Live Paper Trading Book", "⏱️ 3. Historical Backtest"])

# --- TAB 1: LIVE SCANNER ---
with tab1:
    if not st.session_state['chartink_data'].empty:
        df = st.session_state['chartink_data']
        display_df = df[['nsecode', 'name', 'close', 'per_chg', 'volume']].copy()
        display_df.columns = ['Symbol', 'Company', 'LTP', 'Change %', 'Volume']
        st.dataframe(display_df, use_container_width=True)
        
        st.subheader("➕ Add to Paper Trading")
        selected_stocks = st.multiselect("Select stocks to enter tomorrow:", display_df['Symbol'].tolist())
        if st.button("Save to Portfolio"):
            new_trades = []
            for sym in selected_stocks:
                if sym not in st.session_state['portfolio']['Symbol'].values:
                    ltp = display_df[display_df['Symbol'] == sym]['LTP'].values[0]
                    shares = capital // ltp
                    new_trades.append({
                        'Symbol': sym, 'Status': '🟢 OPEN', 'Entry Date': datetime.date.today().strftime('%Y-%m-%d'),
                        'Entry Price': ltp, 'Shares': shares, 'Max Reached': ltp, 
                        'Trailing SL': round(ltp * (1 - sl_percent), 2), 'Current LTP': ltp, 'Unrealized P&L': 0.0
                    })
            if new_trades:
                st.session_state['portfolio'] = pd.concat([st.session_state['portfolio'], pd.DataFrame(new_trades)], ignore_index=True)
                st.session_state['portfolio'].to_csv(PORTFOLIO_FILE, index=False) # <--- AUTO-SAVE PORTFOLIO TO DISK
                st.success(f"Added {len(new_trades)} stocks to your Paper Trading Book!")
        
        if gemini_key:
            st.subheader("🤖 AI Insights (Top 3 Momentum)")
            cols = st.columns(3)
            for idx, row in df.head(3).iterrows():
                with cols[idx % 3]:
                    with st.expander(row['nsecode'].replace('.NS', ''), expanded=True):
                        st.write(get_ai_analysis(row['nsecode'], row['name'], gemini_key))
    else:
        st.info("Run the Daily Scan in the sidebar to view today's breakouts.")

# --- TAB 2: PAPER TRADING PORTFOLIO ---
with tab2:
    st.subheader("📓 Active Paper Portfolio")
    port_df = st.session_state['portfolio']
    
    # HARD BACKUP / RESTORE ROW
    c1, c2 = st.columns([1, 1])
    with c1:
        csv = port_df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Backup", csv, "paper_portfolio.csv", "text/csv")
    with c2:
        restore_file = st.file_uploader("📂 Restore Server Backup (Upload CSV)", type=['csv'], label_visibility="collapsed")
        if restore_file:
            restored_df = pd.read_csv(restore_file)
            st.session_state['portfolio'] = restored_df
            restored_df.to_csv(PORTFOLIO_FILE, index=False) # Save to server disk
            st.rerun() # Refresh UI instantly

    if not port_df.empty:
        if st.button("🔄 Update Live Prices & Trailing SL"):
            with st.spinner("Fetching latest market data..."):
                for idx, row in port_df.iterrows():
                    if row['Status'] == '🟢 OPEN':
                        try:
                            live_data = yf.download(row['Symbol'], period="5d", progress=False)
                            if not live_data.empty:
                                current_close = float(live_data['Close'].iloc[-1])
                                current_high = float(live_data['High'].iloc[-1])
                                
                                if current_high > row['Max Reached']:
                                    port_df.at[idx, 'Max Reached'] = current_high
                                    new_sl = current_high * (1 - sl_percent)
                                    if new_sl > row['Trailing SL']:
                                        port_df.at[idx, 'Trailing SL'] = round(new_sl, 2)
                                
                                port_df.at[idx, 'Current LTP'] = round(current_close, 2)
                                port_df.at[idx, 'Unrealized P&L'] = round((current_close - row['Entry Price']) * row['Shares'], 2)
                                
                                if current_close <= port_df.at[idx, 'Trailing SL']:
                                    port_df.at[idx, 'Status'] = '🔴 CLOSED (SL HIT)'
                        except Exception:
                            pass
                st.session_state['portfolio'] = port_df
                port_df.to_csv(PORTFOLIO_FILE, index=False) # <--- AUTO-SAVE ON UPDATE
                st.success("Prices and Trailing SLs updated!")
                
        def highlight_status(val):
            color = '#00FF88' if 'OPEN' in str(val) else '#FF3366'
            return f'color: {color}'
            
        st.dataframe(port_df.style.map(highlight_status, subset=['Status']), use_container_width=True)
    else:
        st.info("Your portfolio is empty. Select stocks from Tab 1 to start paper trading.")

# --- TAB 3: HISTORICAL RVOL BACKTESTER ---
with tab3:
    st.subheader("⏱️ Time Machine: Top RVOL Breakouts")
    if uploaded_universe is not None:
        uni_df = pd.read_csv(uploaded_universe)
        col_name = next((col for col in uni_df.columns if col.strip().lower() in ['symbol', 'ticker']), uni_df.columns[0])
        hist_tickers = [t.strip() + '.NS' if not t.strip().endswith('.NS') else t.strip() for t in uni_df[col_name].dropna().astype(str).tolist()]
        past_date = st.date_input("Select Historical Scan Date:", max_value=datetime.date.today())
        
        if st.button("Run Historical RVOL Simulation"):
            with st.spinner(f"Simulating market on {past_date}... this takes a minute."):
                target_date = pd.to_datetime(past_date)
                passing_stocks = []
                progress = st.progress(0)
                for idx, t in enumerate(hist_tickers[:100]): 
                    try:
                        data = yf.download(t, period="5y", progress=False)
                        if not data.empty and len(data) > 200:
                            data.index = data.index.tz_localize(None)
                            df_hist = data.loc[:target_date].copy()
                            if len(df_hist) > 200:
                                df_hist = compute_historical_indicators(df_hist)
                                curr, prev = df_hist.iloc[-1], df_hist.iloc[-2]
                                if curr['EMA_10'] > curr['EMA_20'] and prev['EMA_10'] <= prev['EMA_20'] and curr['Close'] > curr['SMA_200']:
                                    passing_stocks.append({'Symbol': t, 'Close': curr['Close'], 'RVOL': curr['RVOL'], 'Data': data})
                    except Exception:
                        pass
                    progress.progress((idx + 1) / len(hist_tickers[:100]))
                progress.empty()
                
                if passing_stocks:
                    results_df = pd.DataFrame(passing_stocks).sort_values(by='RVOL', ascending=False)
                    top_3 = results_df.head(3)
                    st.success(f"Selected Top 3 based on RVOL.")
                    st.dataframe(top_3[['Symbol', 'Close', 'RVOL']].reset_index(drop=True))
                    
                    all_sim_trades = []
                    for _, row in top_3.iterrows():
                        full_data = compute_historical_indicators(row['Data'].copy())
                        future_data = full_data.loc[target_date:] 
                        trades = run_trailing_backtest(future_data, capital, sl_percent)
                        for tr in trades:
                            tr['Symbol'] = row['Symbol']
                            all_sim_trades.append(tr)
                            
                    if all_sim_trades:
                        sim_df = pd.DataFrame(all_sim_trades)[['Symbol', 'Entry Date', 'Exit Date', 'Entry Price', 'Exit Price', 'Shares', 'P&L']]
                        st.metric("Total Strategy P&L", f"₹{sim_df['P&L'].sum():,.2f}")
                        st.dataframe(sim_df, use_container_width=True)
                    else:
                        st.warning("Trades entered, but haven't triggered exit rules yet.")
                else:
                    st.error(f"No EMA breakouts found on {past_date}.")
    else:
        st.info("Please upload your CSV watchlist in the sidebar.")
