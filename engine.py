import numpy as np
import pandas as pd
from scipy.optimize import minimize

#: Nelson-Siegel Yield Curve Model
def nelson_siegel(tau, b0, b1, b2, l1):
    """
    Computes the yield for a given maturity tau.
    b0: Level, b1: Slope, b2: Curvature, l1: Decay factor
    """
    # Prevent division by zero
    l1 = max(l1, 1e-6)
    factor1 = (1 - np.exp(-l1 * tau)) / (l1 * tau)
    factor2 = factor1 - np.exp(-l1 * tau)
    return b0 + (b1 * factor1) + (b2 * factor2)

def fit_yield_curve(maturities, yields):
    """Fits NS parameters to market data using Least Squares."""
    def objective(params):
        b0, b1, b2, l1 = params
        predictions = nelson_siegel(maturities, b0, b1, b2, l1)
        return np.sum((yields - predictions) ** 2)
    
    # Initial guess: Level=Last Yield, Slope=Diff, Curve=0, Decay=0.5
    initial_guess = [yields[-1], yields[0]-yields[-1], 0, 0.5]
    res = minimize(objective, initial_guess, method='Nelder-Mead')
    return res.x

#: Risk Metrics
def calculate_risk_metrics(returns):
    """
    Calculates Volatility and Historical VaR.
    Ensures output is a standard float for formatting.
    """
    # Annualized Volatility 
    vol = returns.std() * np.sqrt(252)
    
    # Historical Value at Risk (95% confidence)
    var_95 = np.percentile(returns, 5)
    
    return float(vol), float(var_95)

from textblob import TextBlob

def get_sentiment(ticker):
    """
    In a production app, you'd scrape live news. 
    Here, we simulate the NLP process by analyzing 'market sentiment' 
    proxies or headlines.
    """
    # Placeholder headlines for demonstration
    # Real-world: Use newsapi.org or yfinance news
    headlines = [
        f"{ticker} shows strong quarterly growth and innovation",
        f"Investors concerned about {ticker} regulatory outlook",
        f"Bullish trend continues for {ticker} as market expands"
    ]
    
    polarities = [TextBlob(h).sentiment.polarity for h in headlines]
    avg_sentiment = sum(polarities) / len(polarities)
    
    if avg_sentiment > 0.1: return "Bullish 📈", avg_sentiment
    elif avg_sentiment < -0.1: return "Bearish 📉", avg_sentiment
    else: return "Neutral ↔️", avg_sentiment