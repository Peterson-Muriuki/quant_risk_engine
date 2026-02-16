import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# Added get_sentiment to the import below
from engine import fit_yield_curve, nelson_siegel, calculate_risk_metrics, get_sentiment

st.set_page_config(page_title="Quant Risk Engine", layout="wide")

# --- UI Header ---
st.title("🛡️ Macro-Sentiment Risk Dashboard")
st.markdown("""
This dashboard combines **Fixed Income Yield Curve Fitting (M1)** with 
**Equity Risk Metrics (M3/Course 9)** and **Sentiment Analysis (M5)**.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker (e.g., AAPL, BTC-USD, ABTC-USD)", "AAPL")
period = st.sidebar.selectbox("Lookback Period", ["1mo", "6mo", "1y", "2y", "5y"], index=2)

# --- Section 1: Yield Curve ---
st.header("1. Government Bond Yield Curve")
with st.expander("View Methodology"):
    st.write("Fitting a Nelson-Siegel model to current US Treasury proxies.")

mats = np.array([0.08, 0.25, 0.5, 1, 2, 5, 10, 30])
yields = np.array([0.0535, 0.0542, 0.0530, 0.0490, 0.0450, 0.0425, 0.0430, 0.0455])

params = fit_yield_curve(mats, yields)
curve_mats = np.linspace(0.1, 30, 100)
fitted_yields = nelson_siegel(curve_mats, *params)

fig = go.Figure()
fig.add_trace(go.Scatter(x=mats, y=yields, mode='markers', name='Market Yields', marker=dict(size=10, color='red')))
fig.add_trace(go.Scatter(x=curve_mats, y=fitted_yields, name='Nelson-Siegel Fit', line=dict(color='blue')))
fig.update_layout(xaxis_title="Maturity (Years)", yaxis_title="Yield (%)", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# --- Section 2: Risk Analysis ---
st.header(f"2. Risk Analysis: {ticker}")

@st.cache_data
def load_data(symbol, range_val):
    df = yf.download(symbol, period=range_val)
    return df

data = load_data(ticker, period)

if not data.empty:
    try:
        close_prices = data['Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]
        
        returns = close_prices.pct_change().dropna()

        if not returns.empty:
            vol, var = calculate_risk_metrics(returns)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Annualized Volatility", f"{vol:.2%}")
            col2.metric("Value at Risk (95%)", f"{var:.2%}")
            col3.metric("Last Close Price", f"${close_prices.iloc[-1]:.2f}")
            
            st.subheader("Daily Returns Distribution")
            st.line_chart(returns)
            
            # --- Section 3: Alternative Data - Sentiment Analysis (M5) ---
            st.divider()
            st.header("3. Alternative Data: Sentiment Analysis")
            sentiment_label, score = get_sentiment(ticker)

            s_col1, s_col2 = st.columns(2)
            s_col1.metric("Market Sentiment", sentiment_label)
            s_col2.metric("Sentiment Score (NLP)", f"{score:.2f}")

            st.info(f"The NLP Engine (M5) has analyzed simulated headlines for {ticker}. "
                    f"A score > 0 indicates positive momentum.")
            
        else:
            st.warning("Insufficient data to calculate returns.")
    except Exception as e:
        st.error(f"Error processing data: {e}")
else:
    st.error("No data found. Please check the ticker symbol.")

st.divider()
st.caption("Data source: Yahoo Finance. Yields are illustrative proxies.")