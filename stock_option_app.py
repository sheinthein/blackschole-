
import streamlit as st
import yfinance as yf
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# Fetch stock data
def fetch_stock_data(ticker_symbol, period='1y', interval='1d'):
    ticker = yf.Ticker(ticker_symbol)
    stock_data = ticker.history(period=period, interval=interval)
    return stock_data

# Calculate historical volatility
def calculate_historical_volatility(stock_data, window=252):
    log_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    volatility = np.sqrt(window) * log_returns.std()
    return volatility

# Black-Scholes Model class
class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = S        # Underlying asset price
        self.K = K        # Option strike price
        self.T = T        # Time to expiration in years
        self.r = r        # Risk-free interest rate
        self.sigma = sigma  # Volatility of the underlying asset

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def call_option_price(self):
        return (self.S * si.norm.cdf(self.d1(), 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0.0, 1.0))
    
    def put_option_price(self):
        return (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0.0, 1.0) - self.S * si.norm.cdf(-self.d1(), 0.0, 1.0))

# Black-Scholes Greeks class
class BlackScholesGreeks(BlackScholesModel):
    def delta_call(self):
        return si.norm.cdf(self.d1(), 0.0, 1.0)

    def delta_put(self):
        return -si.norm.cdf(-self.d1(), 0.0, 1.0)
    
    def gamma(self):
        return si.norm.pdf(self.d1(), 0.0, 1.0) / (self.S * self.sigma * np.sqrt(self.T))

    def theta_call(self):
        return (-self.S * si.norm.pdf(self.d1(), 0.0, 1.0) * self.sigma / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0.0, 1.0))
    
    def theta_put(self):
        return (-self.S * si.norm.pdf(self.d1(), 0.0, 1.0) * self.sigma / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0.0, 1.0))

    def vega(self):
        return self.S * si.norm.pdf(self.d1(), 0.0, 1.0) * np.sqrt(self.T)
    
    def rho_call(self):
        return self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0.0, 1.0)
    
    def rho_put(self):
        return -self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0.0, 1.0)

# Streamlit app begins here
st.title('Stock Option Pricing with Black-Scholes Model')

# Inputs
ticker_symbol = st.text_input("Enter the stock ticker symbol (e.g., 'AAPL'):", 'AAPL')
period = st.selectbox("Select the time period for stock data:", ['1y', '6mo', '3mo', '1mo'])
interval = st.selectbox("Select the data interval:", ['1d', '1wk', '1mo'])
stock_data = fetch_stock_data(ticker_symbol, period, interval)

# Function to fetch the current stock price
def current_price(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    stock_info = ticker.history(period="1d")
    return stock_info['Close'].iloc[-1] if not stock_info.empty else None

# Fetch the current price
current_price_value = current_price(ticker_symbol)

if stock_data.empty:
    st.error("No stock data found. Please enter a valid ticker symbol.")
else:
    st.write(f"The current price of {ticker_symbol} is: ${current_price_value:.2f}")
    st.write(f"Displaying {ticker_symbol} historical stock price for the past {period}.")
    
    # Plotting stock data
    st.line_chart(stock_data['Close'])

    # Fetching and displaying historical volatility
    hist_volatility = calculate_historical_volatility(stock_data)
    st.write(f"Historical Volatility: {hist_volatility:.2f}")

    # User inputs for Black-Scholes parameters
    S = current_price_value  # Fetch the current stock price
    K = st.number_input("Enter the strike price of the option:", value=100.0)
    T = st.number_input("Enter the time to expiration (in years):", value=1.0)
    r = st.number_input("Enter the risk-free interest rate (as a decimal, e.g., 0.05 for 5%):", value=0.05)
    sigma = st.number_input("Enter the volatility (as a decimal, e.g., 0.2 for 20%):", value=hist_volatility)

    # Black-Scholes calculations
    bsm = BlackScholesModel(S=S, K=K, T=T, r=r, sigma=sigma)
    call_price = bsm.call_option_price()
    put_price = bsm.put_option_price()

    st.write(f"Call Option Price: ${call_price:.2f}")
    st.write(f"Put Option Price: ${put_price:.2f}")

    # Greeks calculations
    bsg = BlackScholesGreeks(S=S, K=K, T=T, r=r, sigma=sigma)
    st.write(f"Call Delta: {bsg.delta_call():.2f}")
    st.write(f"Put Delta: {bsg.delta_put():.2f}")
    
    # Define a range of stock prices and plot the delta of a call option
    stock_prices = np.linspace(S * 0.8, S * 1.2, 100)
    deltas = [BlackScholesGreeks(S=price, K=K, T=T, r=r, sigma=sigma).delta_call() for price in stock_prices]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock_prices, deltas)
    ax.set_title(f'Delta of a Call Option as {ticker_symbol} Price Changes')
    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Delta')
    st.pyplot(fig)

    # Create heatmap for option pricing
    st.subheader(f"Expected value of Option returns for {ticker_symbol}")
    
    # Define ranges for underlying prices and strike prices
    underlying_prices = np.linspace(S * 0.9, S * 1.1, 50)  # Simulating prices from 90% to 110% of current price
    strike_prices = np.linspace(K * 0.9, K * 1.1, 50)      # Simulating strike prices from 90% to 110% of current strike

    # Create a matrix to store option prices
    option_values = np.zeros((len(strike_prices), len(underlying_prices)))

    # Populate the option values matrix
    for i, strike in enumerate(strike_prices):
        for j, price in enumerate(underlying_prices):
            bsm_temp = BlackScholesModel(S=price, K=strike, T=T, r=r, sigma=sigma)
            option_values[i, j] = bsm_temp.call_option_price()

    # Plot the heatmap using Plotly
    fig = px.imshow(option_values,
                    x=underlying_prices,
                    y=strike_prices,
                    color_continuous_scale='RdYlGn',
                    labels=dict(x="Underlying Price", y="Option Strike Price", color="Call Option Value"))
    fig.update_layout(title=f"Expected value of Option returns for {ticker_symbol}")
    st.plotly_chart(fig)
