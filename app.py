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
st.set_page_config(page_title="Quant Momentum + AI", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #0B0E14; color: #E2E8F0; }
    .st-emotion-cache-1wivap2, .st-emotion-cache-16txtl3 { padding: 2rem 1rem; }
    h1, h2, h3 { color: #FFFFFF; font-family: 'Inter', sans-serif; }
    
    .stButton>button {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        color: #000000;
        font-weight: 800;
        border-radius: 6px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4); }
    
    div[data-testid="stMetricValue"] { color: #00FF88 !important; font-size: 2rem !important; }
    div[data-testid="stMetricLabel"] { color: #94A3B8 !important; font-weight: 600; }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Quant Momentum & AI Engine")
st.markdown("Automated Chartink Scraping, Trailing Risk Backtester & Gemini Trade Analyst")

# ==========================================
# 2. GEMINI AI ANALYST FUNCTION
# ==========================================
def get_ai_analysis(ticker, company_name, api_key):
    """Passes the breakout data to Gemini for a rapid technical/fundamental narrative."""
    if not api_key:
        return "⚠️ Please enter your Gemini API key in the sidebar to view AI insights."
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
        You are an expert quantitative trader. The Indian stock {ticker} ({company_name}) just appeared on our high-momentum Chartink volume breakout scanner today.
        In exactly 3 bullet points, provide a rapid risk-reward analysis of this setup. 
        Mention any known macroeconomic sector tailwinds for this industry. Be highly concise and professional.
        """
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e:
        return f"AI Analysis failed: {e}"

# ==========================================
# 3. CHARTINK SCRAPER FUNCTION
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def scrape_chartink(screener_url, manual_clause=""):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        with requests.Session() as s:
            r = s.get(screener_url, headers=headers)
            soup = BeautifulSoup(r.text, 'html.parser')
            
            csrf_meta = soup.find('meta', {'name': 'csrf-token'})
            csrf = csrf_meta['content'] if csrf_meta else ""
            
            scan_clause = manual_clause
            if not scan_clause:
                clause_match = re.search(r'scan_clause\s*:\s*\'(.*?)\'', r.text)
                if not clause_match:
                     clause_match = re.search(r'name="scan_clause"\s+value="(.*?)"', r.text)
                scan_clause = clause_match.group(1) if clause_match else ""

            if not scan_clause:
                return pd.DataFrame(), "Could not auto-detect scan clause. Please paste it manually."

            process_url = 'https://chartink.com/screener/process'
            post_headers = {'x-csrf-token': csrf, 'X-Requested-With': 'XMLHttpRequest', **headers}
            payload = {'scan_clause': scan_clause}
            
            res = s.post(process_url, data=payload, headers=post_headers)
            if res.status_code == 200:
                data = res.json()
                if 'data' in data and len(data['data']) > 0:
                    df = pd.DataFrame(data['data'])
                    df['nsecode'] = df['nsecode'] + '.NS'
                    return df, "Success"
            return pd.DataFrame(), f"Backend rejected request. Status: {res.status_code}"
    except Exception as e:
        return pd.DataFrame(), str(e)

# ==========================================
# 4. TRAILING RISK BACKTEST ENGINE
# ==========================================
def run_trailing_backtest(df_price, capital=100000, sl_pct=0.02):
    trades = []
    in_pos, entry_price, entry_date, shares, highest_price, current_sl = False, 0, None, 0, 0, 0
    df_price['EMA_20'] = df_price['Close'].ewm(span=20, adjust=False).mean()
    
    for i in range(1, len(df_price)):
        curr, prev = df_price.iloc[i], df_price.iloc[i-1]
        
        # ENTRY
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
                profit_loss = (exit_price - entry_price) * shares
                roi_pct = ((exit_price - entry_price) / entry_price) * 100
                trades.append({
                    "Entry Date": entry_date.strftime('%Y-%m-%d'),
                    "Exit Date": df_price.index[i].strftime('%Y-%m-%d'),
                    "Entry": round(entry_price, 2), "Exit": round(exit_price, 2),
                    "Shares": int(shares), "Max Reached": round(highest_price, 2),
                    "P&L (₹)": round(profit_loss, 2), "ROI %": round(roi_pct, 2)
                })
    return pd.DataFrame(trades)

# ==========================================
# 5. DASHBOARD LAYOUT & EXECUTION
# ==========================================
with st.sidebar:
    st.header("⚙️ Scanner Config")
    url_input = st.text_input("Chartink Screener URL", value="https://chartink.com/screener/momentum-stocks-29112990")
    
    with st.expander("Risk Settings"):
        capital = st.number_input("Capital Per Stock (₹)", value=100000, step=10000)
        sl_percent = st.number_input("Trailing Stop Loss (%)", value=2.0, step=0.5) / 100
        manual_clause = st.text_area("Manual Scan Clause (Optional)")
        
    st.header("🤖 AI Setup")
    gemini_key = st.text_input("Gemini API Key", type="password", placeholder="Enter key for AI Insights")
        
    run_scan = st.button("🚀 EXECUTE SCAN")

if run_scan:
    with st.spinner("Bypassing Chartink & Extracting Live Data..."):
        chartink_df, status = scrape_chartink(url_input, manual_clause)
        
    if not chartink_df.empty:
        st.success(f"Successfully scraped {len(chartink_df)} stocks from Chartink!")
        
        display_df = chartink_df[['sr', 'nsecode', 'name', 'close', 'per_chg', 'volume']].copy()
        display_df.columns = ['#', 'Symbol', 'Company', 'LTP', 'Change %', 'Volume']
        st.dataframe(display_df.set_index('#'), use_container_width=True)
        
        # --- GEMINI AI ANALYST SECTION ---
        st.markdown("---")
        st.subheader("🤖 Gemini AI Analyst: Top 3 Breakout Profiles")
        if gemini_key:
            top_3_stocks = chartink_df.head(3)
            ai_cols = st.columns(3)
            
            for idx, row in top_3_stocks.iterrows():
                col = ai_cols[idx % 3]
                ticker = row['nsecode'].replace('.NS', '')
                company = row['name']
                
                with col:
                    with st.expander(f"Deep Dive: {ticker}", expanded=True):
                        with st.spinner("Analyzing chart..."):
                            analysis = get_ai_analysis(ticker, company, gemini_key)
                            st.write(analysis)
        else:
            st.warning("⚠️ Enter your Gemini API Key in the sidebar to unlock AI sector and risk analysis.")

        # --- RUN BACKTEST ON RESULTS ---
        st.markdown("---")
        st.subheader("📊 1 Lakh Capital Backtest (Trailing 2% SL)")
        st.markdown(f"*Simulating historical momentum entries on today's passing stocks. Capital: ₹{capital:,.0f} per trade.*")
        
        tickers = chartink_df['nsecode'].tolist()
        all_trades = pd.DataFrame()
        
        progress_bar = st.progress(0)
        for idx, ticker in enumerate(tickers):
            try:
                stock_data = yf.download(ticker, period="1y", interval="1d", progress=False)
                if not stock_data.empty:
                    if isinstance(stock_data.columns, pd.MultiIndex):
                        stock_data.columns = stock_data.columns.droplevel(1)
                    trade_history = run_trailing_backtest(stock_data, capital=capital, sl_pct=sl_percent)
                    if not trade_history.empty:
                        trade_history.insert(0, 'Symbol', ticker.replace('.NS', ''))
                        all_trades = pd.concat([all_trades, trade_history])
            except Exception:
                pass
            progress_bar.progress((idx + 1) / len(tickers))
        progress_bar.empty()
        
        if not all_trades.empty:
            all_trades = all_trades.sort_values(by='Exit Date', ascending=False).reset_index(drop=True)
            
            total_pnl = all_trades['P&L (₹)'].sum()
            win_rate = (len(all_trades[all_trades['P&L (₹)'] > 0]) / len(all_trades)) * 100
            best_trade = all_trades['P&L (₹)'].max()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total P&L Generated", f"₹{total_pnl:,.2f}")
            c2.metric("Win Rate", f"{win_rate:.1f}%")
            c3.metric("Total Trades", len(all_trades))
            c4.metric("Best Single Trade", f"₹{best_trade:,.2f}")
            
            all_trades['Cumulative P&L'] = all_trades['P&L (₹)'].cumsum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=all_trades.index, y=all_trades['Cumulative P&L'], mode='lines', fill='tozeroy', line=dict(color='#00FF88', width=3)))
            fig.update_layout(title="Portfolio Equity Curve (₹)", template="plotly_dark", plot_bgcolor='#0B0E14', paper_bgcolor='#0B0E14', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(all_trades, use_container_width=True)
        else:
            st.info("No completed trades triggered in the backtest period.")
            
    else:
        st.error(f"Failed to scrape Chartink. Error: {status}")
