import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import re
from google import genai

# ==========================================
# 1. UI OVERHAUL: MODERN FINTECH DASHBOARD
# ==========================================
st.set_page_config(page_title="Quant Momentum + AI", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0B0E14; color: #E2E8F0; }
    h1, h2, h3 { color: #FFFFFF; font-family: 'Inter', sans-serif; }
    .stButton>button { background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%); color: #000; font-weight: bold; border: none; }
    div[data-testid="stMetricValue"] { color: #00FF88 !important; font-size: 2rem !important; }
    div[data-testid="stMetricLabel"] { color: #94A3B8 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Quant Momentum & AI Engine")

# --- INITIALIZE SESSION STATE FOR SCRAPED DATA ---
if 'chartink_data' not in st.session_state:
    st.session_state['chartink_data'] = pd.DataFrame()

# ==========================================
# 2. AI & SCRAPING FUNCTIONS
# ==========================================
def get_ai_analysis(ticker, company_name, api_key):
    if not api_key: return "⚠️ Please enter your Gemini API key in the sidebar."
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"You are a quant trader. The stock {ticker} ({company_name}) hit our momentum screener. Give 3 short bullet points on risk-reward and sector tailwinds."
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
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
                clause_match = re.search(r'scan_clause\s*:\s*\'(.*?)\'', r.text) or re.search(r'name="scan_clause"\s+value="(.*?)"', r.text)
                scan_clause = clause_match.group(1) if clause_match else ""

            if not scan_clause: return pd.DataFrame(), "Could not auto-detect scan clause."

            res = s.post('https://chartink.com/screener/process', 
                         data={'scan_clause': scan_clause}, 
                         headers={'x-csrf-token': csrf, 'X-Requested-With': 'XMLHttpRequest', **headers})
            
            if res.status_code == 200:
                data = res.json()
                if 'data' in data and len(data['data']) > 0:
                    df = pd.DataFrame(data['data'])
                    df['nsecode'] = df['nsecode'] + '.NS'
                    return df, "Success"
            return pd.DataFrame(), f"Backend rejected request: {res.status_code}"
    except Exception as e:
        return pd.DataFrame(), str(e)

# ==========================================
# 3. TRAILING RISK BACKTEST ENGINE
# ==========================================
def run_trailing_backtest(df_price, capital, sl_pct):
    trades = []
    in_pos, entry_price, entry_date, shares, highest_price, current_sl = False, 0, None, 0, 0, 0
    df_price['EMA_20'] = df_price['Close'].ewm(span=20, adjust=False).mean()
    
    for i in range(1, len(df_price)):
        curr, prev = df_price.iloc[i], df_price.iloc[i-1]
        
        # ENTRY (Momentum crossover proxy)
        if not in_pos and curr['Close'] > curr['EMA_20'] and prev['Close'] <= prev['EMA_20']:
            in_pos, entry_price, entry_date = True, curr['Close'], df_price.index[i]
            shares = capital // entry_price
            highest_price, current_sl = entry_price, entry_price * (1 - sl_pct)
            
        # TRAIL & EXIT
        elif in_pos:
            if curr['High'] > highest_price:
                highest_price = curr['High']
                new_sl = highest_price * (1 - sl_pct)
                if new_sl > current_sl: current_sl = new_sl 
            
            if curr['Low'] <= current_sl:
                in_pos = False
                exit_price = current_sl
                trades.append({
                    "Status": "Closed", "Entry Date": entry_date.strftime('%Y-%m-%d'), "Exit Date": df_price.index[i].strftime('%Y-%m-%d'),
                    "Entry": round(entry_price, 2), "Exit": round(exit_price, 2), "Shares": int(shares), 
                    "Max Reached": round(highest_price, 2), "P&L (₹)": round((exit_price - entry_price) * shares, 2),
                    "ROI %": round(((exit_price - entry_price) / entry_price) * 100, 2)
                })
    
    # INCLUDE OPEN TRADES (If stopped out hasn't happened yet)
    if in_pos:
        curr = df_price.iloc[-1]
        trades.append({
            "Status": "🟢 OPEN", "Entry Date": entry_date.strftime('%Y-%m-%d'), "Exit Date": "Running...",
            "Entry": round(entry_price, 2), "Exit": round(curr['Close'], 2), "Shares": int(shares), 
            "Max Reached": round(highest_price, 2), "P&L (₹)": round((curr['Close'] - entry_price) * shares, 2),
            "ROI %": round(((curr['Close'] - entry_price) / entry_price) * 100, 2)
        })
        
    return pd.DataFrame(trades)

# ==========================================
# 4. SIDEBAR CONFIG
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings")
    url_input = st.text_input("Chartink URL", value="https://chartink.com/screener/momentum-stocks-29112990")
    capital = st.number_input("Capital Per Stock (₹)", value=100000, step=10000)
    sl_percent = st.number_input("Trailing SL / Target (%)", value=2.0, step=0.5) / 100
    gemini_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    st.header("🔧 Bypass Chartink Block")
    st.markdown("Chartink occasionally blocks cloud servers. To bypass, paste the **scan_clause** from the website's source code below:")
    
    # Here is the missing input box!
    manual_clause = st.text_area("Manual Scan Clause", placeholder="( {33619} ( latest close > ... ) )")
    
    if st.button("🚀 PULL CHARTINK DATA"):
        with st.spinner("Scraping Chartink..."):
            # I also updated this line to ensure the manual_clause is passed to the scraper
            df, status = scrape_chartink(url_input, manual_clause) 
            if not df.empty:
                st.session_state['chartink_data'] = df
                st.success(f"Scraped {len(df)} stocks!")
            else:
                st.error(f"Failed: {status}")

# ==========================================
# 5. TABS & UI EXECUTION
# ==========================================
tab1, tab2 = st.tabs(["📊 1. Live Chartink & AI Analysis", "🧪 2. Trailing Target Backtester"])

# --- TAB 1: LIVE SCANNER ---
with tab1:
    if not st.session_state['chartink_data'].empty:
        df = st.session_state['chartink_data']
        display_df = df[['sr', 'nsecode', 'name', 'close', 'per_chg', 'volume']].copy()
        display_df.columns = ['#', 'Symbol', 'Company', 'LTP', 'Change %', 'Volume']
        st.dataframe(display_df.set_index('#'), use_container_width=True)
        
        if gemini_key:
            st.subheader("🤖 AI Insights (Top 3)")
            cols = st.columns(3)
            for idx, row in df.head(3).iterrows():
                with cols[idx % 3]:
                    with st.expander(row['nsecode'].replace('.NS', ''), expanded=True):
                        st.write(get_ai_analysis(row['nsecode'], row['name'], gemini_key))
    else:
        st.info("👈 Click 'PULL CHARTINK DATA' in the sidebar to begin.")

# --- TAB 2: BACKTESTER ---
with tab2:
    if not st.session_state['chartink_data'].empty:
        st.subheader(f"Historical Momentum Backtest (₹{capital:,.0f} per stock | 2% Trailing Target)")
        
        test_ticker = st.selectbox("Select Scraped Stock to Test", st.session_state['chartink_data']['nsecode'].tolist())
        
        if st.button("Run Trailing Backtest on Stock"):
            with st.spinner("Fetching historical data and trailing targets..."):
                stock_data = yf.download(test_ticker, period="1y", interval="1d", progress=False)
                if isinstance(stock_data.columns, pd.MultiIndex): stock_data.columns = stock_data.columns.droplevel(1)
                
                trade_history = run_trailing_backtest(stock_data, capital, sl_percent)
                
                if not trade_history.empty:
                    # Metrics
                    closed_trades = trade_history[trade_history['Status'] == 'Closed']
                    win_rate = (len(closed_trades[closed_trades['P&L (₹)'] > 0]) / len(closed_trades)) * 100 if len(closed_trades) > 0 else 0
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Trades", len(trade_history))
                    c2.metric("Win Rate (Closed Trades)", f"{win_rate:.1f}%")
                    c3.metric("Net P&L", f"₹{trade_history['P&L (₹)'].sum():,.2f}")
                    
                    st.dataframe(trade_history.style.applymap(lambda x: 'color: #00FF88' if x == '🟢 OPEN' else '', subset=['Status']), use_container_width=True)
                else:
                    st.warning("No entries triggered for this stock in the last year.")
    else:
        st.info("👈 Pull Chartink data first to enable backtesting.")
