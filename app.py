import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from google import genai
import datetime
import os
import json

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

st.title("⚡ Autonomous Quant & AI Paper Trading Engine")

SCAN_FILE = "latest_scan.csv"
PORTFOLIO_FILE = "portfolio_backup.csv"
PORTFOLIO_COLS = ['Symbol', 'Status', 'Entry Date', 'Entry Price', 'Shares', 'Max Reached', 'Trailing SL', 'Current LTP', 'Unrealized P&L', 'AI Score', 'AI Thesis']

# Initialize or Load States
if 'chartink_data' not in st.session_state:
    st.session_state['chartink_data'] = pd.read_csv(SCAN_FILE) if os.path.exists(SCAN_FILE) else pd.DataFrame()

if 'portfolio' not in st.session_state:
    if os.path.exists(PORTFOLIO_FILE):
        df = pd.read_csv(PORTFOLIO_FILE)
        for col in PORTFOLIO_COLS:
            if col not in df.columns: df[col] = ""
        st.session_state['portfolio'] = df
    else:
        st.session_state['portfolio'] = pd.DataFrame(columns=PORTFOLIO_COLS)

# ==========================================
# 2. CORE FUNCTIONS
# ==========================================
def get_batch_ai_scores(dataframe, api_key):
    """Sends all top stocks to Gemini in a SINGLE API call to avoid 429 Rate Limits."""
    if not api_key: 
        st.error("No API Key provided.")
        return dataframe
        
    try:
        client = genai.Client(api_key=api_key)
        
        # Format list for Gemini
        stock_list = "\n".join([f"- {row['nsecode']}: {row['name']}" for _, row in dataframe.iterrows()])
        
        prompt = f"""
        You are a strict institutional quant analyst. Evaluate these Indian stocks for a momentum breakout tomorrow:
        {stock_list}
        
        Respond STRICTLY in JSON format. Do not use markdown blocks. 
        Format exactly like this:
        {{
            "RELIANCE.NS": {{"score": 85, "thesis": "Strong volume, sector tailwinds..."}},
            "TCS.NS": {{"score": 60, "thesis": "IT sector weak, wait for pullback..."}}
        }}
        """
        
        res = client.models.generate_content(model='gemini-2.5-flash', contents=prompt).text
        
        # Clean and parse JSON
        clean_json = res.replace("```json", "").replace("```", "").strip()
        scores = json.loads(clean_json)
        
        # Map scores back to the dataframe
        for idx, row in dataframe.iterrows():
            symbol = row['nsecode']
            if symbol in scores:
                dataframe.at[idx, 'AI_Score'] = scores[symbol].get('score', 0)
                dataframe.at[idx, 'AI_Analysis'] = scores[symbol].get('thesis', "No thesis provided.")
        return dataframe
    except Exception as e:
        st.error(f"Batch AI Error: {e}")
        return dataframe

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

            res = s.post('https://chartink.com/screener/process', data={'scan_clause': scan_clause}, headers={'x-csrf-token': csrf, 'X-Requested-With': 'XMLHttpRequest', **headers})
            
            if res.status_code == 200:
                data = res.json()
                if 'error' in data: return pd.DataFrame(), f"Chartink Error: {data['error']}"
                if 'data' in data:
                    if len(data['data']) == 0: return pd.DataFrame(), "Success, but 0 stocks matched today."
                    df = pd.DataFrame(data['data'])
                    df['nsecode'] = df['nsecode'] + '.NS'
                    df['AI_Score'] = 0
                    df['AI_Analysis'] = "Pending AI Scan..."
                    return df, "Success"
            return pd.DataFrame(), f"Rejected: {res.status_code}"
    except Exception as e:
        return pd.DataFrame(), str(e)

# ==========================================
# 3. SIDEBAR CONFIG & SECRETS
# ==========================================
with st.sidebar:
    st.header("⚙️ Execution Settings")
    capital = st.number_input("Capital Per Trade (₹)", value=100000, step=10000)
    sl_percent = st.number_input("Trailing Risk (%)", value=2.0, step=0.5) / 100
    
    secret_api = st.secrets.get("GEMINI_API_KEY", "")
    gemini_key = secret_api if secret_api else st.text_input("Gemini API Key", type="password")
    if secret_api: st.success("✅ Gemini Key loaded.")
    
    st.markdown("---")
    st.header("🔍 Chartink Scanner")
    secret_url = st.secrets.get("CHARTINK_URL", "https://chartink.com/screener/momentum-stocks-29112990")
    secret_clause = st.secrets.get("CHARTINK_CLAUSE", "")
    url_input = st.text_input("URL", value=secret_url)
    manual_clause = secret_clause if secret_clause else st.text_area("Manual Scan Clause (Bypass)")
    if secret_clause: st.success("✅ Scan Clause loaded.")
        
    if st.button("🚀 RUN EOD SCAN"):
        with st.spinner("Scraping EOD data..."):
            df, status = scrape_chartink(url_input, manual_clause)
            if not df.empty:
                st.session_state['chartink_data'] = df
                df.to_csv(SCAN_FILE, index=False)
                st.success(f"Scraped {len(df)} stocks!")
            else:
                st.error(status)

# ==========================================
# 4. TABBED INTERFACE
# ==========================================
tab1, tab2 = st.tabs(["⚡ 1. EOD Scan & Deep AI Scoring", "📓 2. Live Portfolio (9:15 AM Execution)"])

# --- TAB 1: EOD SCANNER & AI ---
with tab1:
    if not st.session_state['chartink_data'].empty:
        df = st.session_state['chartink_data']
        
        st.subheader("1. Run Deep AI Analysis")
        st.write("Score today's breakouts using Gemini before adding them to your trading book.")
        
        if st.button("🧠 Generate AI Scores (Batch Process)"):
            if gemini_key:
                with st.spinner("Sending batch request to Gemini..."):
                    # Only analyze top 15 by volume to ensure we stay well within API limits
                    top_stocks = df.sort_values(by="volume", ascending=False).head(15)
                    scored_df = get_batch_ai_scores(top_stocks, gemini_key)
                    
                    # Merge scores back into main dataframe
                    for idx, row in scored_df.iterrows():
                        df.at[idx, 'AI_Score'] = row['AI_Score']
                        df.at[idx, 'AI_Analysis'] = row['AI_Analysis']
                        
                st.session_state['chartink_data'] = df
                df.to_csv(SCAN_FILE, index=False)
                st.success("Deep AI Analysis Complete! No API Limits Hit.")
            else:
                st.error("Please provide a Gemini API Key in the sidebar.")

        st.markdown("---")
        display_df = df[['nsecode', 'name', 'close', 'volume', 'AI_Score', 'AI_Analysis']].copy()
        display_df.columns = ['Symbol', 'Company', 'LTP', 'Volume', 'AI Score', 'AI Thesis']
        st.dataframe(display_df.sort_values(by="AI Score", ascending=False), use_container_width=True)
        
        st.markdown("---")
        st.subheader("2. Add to Next-Day Execution Book (9:15 AM)")
        
        selected_stocks = st.multiselect("Select High-Conviction Stocks:", display_df['Symbol'].tolist())
        if st.button("Log Trades for Tomorrow"):
            new_trades = []
            for sym in selected_stocks:
                if sym not in st.session_state['portfolio']['Symbol'].values:
                    row_data = display_df[display_df['Symbol'] == sym].iloc[0]
                    ltp = row_data['LTP']
                    shares = capital // ltp
                    next_day = (datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                    
                    new_trades.append({
                        'Symbol': sym, 'Status': '🟢 OPEN (9:15 AM)', 'Entry Date': next_day,
                        'Entry Price': ltp, 'Shares': shares, 'Max Reached': ltp, 
                        'Trailing SL': round(ltp * (1 - sl_percent), 2), 'Current LTP': ltp, 'Unrealized P&L': 0.0,
                        'AI Score': row_data['AI Score'], 'AI Thesis': row_data['AI Thesis']
                    })
            if new_trades:
                st.session_state['portfolio'] = pd.concat([st.session_state['portfolio'], pd.DataFrame(new_trades)], ignore_index=True)
                st.session_state['portfolio'].to_csv(PORTFOLIO_FILE, index=False)
                st.success(f"Added {len(new_trades)} stocks to your Paper Trading Book!")
    else:
        st.info("Run the EOD Scan in the sidebar to view today's breakouts.")

# --- TAB 2: PAPER TRADING PORTFOLIO ---
with tab2:
    st.subheader("📓 Live Portfolio Manager (-2% Trailing Target)")
    port_df = st.session_state['portfolio']
    
    c1, c2 = st.columns([1, 1])
    with c1:
        csv = port_df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Permanent Backup", csv, "paper_portfolio.csv", "text/csv")
    with c2:
        restore_file = st.file_uploader("📂 Restore Server Backup (Upload CSV)", type=['csv'], label_visibility="collapsed")
        if restore_file:
            restored_df = pd.read_csv(restore_file)
            st.session_state['portfolio'] = restored_df
            restored_df.to_csv(PORTFOLIO_FILE, index=False)
            st.rerun()

    if not port_df.empty:
        if st.button("🔄 Update Live Market Prices"):
            with st.spinner("Fetching latest market data & adjusting trailing stops..."):
                for idx, row in port_df.iterrows():
                    if 'OPEN' in str(row['Status']):
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
                port_df.to_csv(PORTFOLIO_FILE, index=False)
                st.success("Prices and Risk Metrics updated!")
                
        def highlight_status(val):
            if 'OPEN' in str(val): return 'color: #00FF88; font-weight: bold;'
            if 'SL HIT' in str(val): return 'color: #FF3366; font-weight: bold;'
            return ''
            
        st.dataframe(
            port_df.style.map(highlight_status, subset=['Status']), 
            use_container_width=True,
            column_config={
                "AI Score": st.column_config.NumberColumn("AI Score", help="0-100 Conviction Score", format="%d ⭐"),
                "AI Thesis": st.column_config.TextColumn("AI Thesis", help="Hover to read full deep analysis", width="large")
            }
        )
    else:
        st.info("Your portfolio is empty. Run an EOD scan and add high-conviction trades to begin.")
