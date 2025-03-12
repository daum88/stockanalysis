import yfinance as yf
import pandas as pd
import numpy as np
import talib
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Function to fetch news sentiment for a given stock ticker using News API
def get_stock_news_sentiment(ticker):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    from_date = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')

    try:
        print(f"Fetching news sentiment for {ticker}...")
        # API request to fetch stock-related news
        news_api_url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&from={from_date}&apiKey={API_KEY}"
        response = requests.get(news_api_url).json()

        # Handle API errors
        if response.get("status") == "error":
            print(f"News API error: {response.get('message')}")
            return 0.0

        # Extract headlines and calculate sentiment scores
        headlines = [article['title'] for article in response.get("articles", [])[:10]]
        sentiment_scores += [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    except Exception as e:
        print(f"Error fetching news sentiment for {ticker}: {e}")
        return 0.0

    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
    return avg_sentiment

# Function to fetch stock data and calculate financial & technical indicators
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        print(f"Fetching stock data for {ticker}...")
        info = stock.info  # Get stock details
        df = stock.history(period="1y")  # Fetch historical stock data for 1 year

        if df.empty:
            print(f"Skipping {ticker}: No data returned from Yahoo Finance.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

    # Compute Technical Indicators
    df['RSI'] = talib.RSI(df['Close'])  # Relative Strength Index (RSI)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'])  # Moving Average Convergence Divergence (MACD)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)  # 50-day Simple Moving Average
    df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)  # 200-day Simple Moving Average

    # Get the latest stock data
    latest_data = df.iloc[-1]
    news_sentiment = get_stock_news_sentiment(ticker)  # Fetch news sentiment
    eps = info.get("trailingEps")  # Earnings per Share
    revenue_growth = info.get("revenueGrowth")  # Revenue Growth Rate
    discount_rate = 0.08  # Assumed discount rate for DCF valuation

    # Calculate a rough fair value estimate using Discounted Cash Flow (DCF) model
    if eps and revenue_growth and eps > 0:
        projected_eps = eps * (1 + revenue_growth) ** 5
        fair_value = projected_eps / ((1 + discount_rate) ** 5)
        if fair_value < (info.get("currentPrice") * 0.1):
            fair_value = None
    else:
        fair_value = None

    # Calculate undervaluation percentage
    undervaluation_pct = ((fair_value - info.get("currentPrice")) / fair_value) * 100 if fair_value else None

    # Compile extracted stock data
    data = {
        "Ticker": ticker,
        "Current Price": info.get("currentPrice"),
        "P/E Ratio": info.get("forwardPE"),
        "EPS": eps,
        "Market Cap": info.get("marketCap"),
        "52w High": info.get("fiftyTwoWeekHigh"),
        "52w Low": info.get("fiftyTwoWeekLow"),
        "Beta": info.get("beta"),
        "Dividend Yield": info.get("dividendYield"),
        "Revenue Growth": revenue_growth,
        "Sentiment Score": news_sentiment,
        "Fair Value Estimate (DCF)": fair_value,
        "Undervaluation (%)": undervaluation_pct,
        "RSI": latest_data['RSI'],
        "MACD": latest_data['MACD'],
        "SMA_50": latest_data['SMA_50'],
        "SMA_200": latest_data['SMA_200']
    }
    return pd.DataFrame([data])

# Function to analyze 50 random stock tickers
def analyze_stocks():
    def get_sp500_tickers():
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        return tables[0]["Symbol"].tolist()

    tickers = random.sample(get_sp500_tickers(), 50)
    stock_data = [get_stock_data(ticker) for ticker in tickers if not get_stock_data(ticker).empty]

    if not stock_data:
        print("No valid stock data collected.")
        return pd.DataFrame()

    return pd.concat(stock_data, ignore_index=True)

# Function to train a Machine Learning model for stock price prediction
def train_ml_model(stock_results, model_type="RandomForest"):
    stock_results = stock_results.dropna()
    features = stock_results[["P/E Ratio", "EPS", "Market Cap", "52w High", "52w Low", "Beta", "Dividend Yield", "Revenue Growth", "Sentiment Score", "RSI", "MACD", "SMA_50", "SMA_200"]]
    target = stock_results["Current Price"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)

    # Choose the model type (RandomForest or Neural Network)
    if model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    elif model_type == "NeuralNetwork":
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {error}")

    return model

# Execute stock analysis and machine learning training
if __name__ == "__main__":
    print("Starting stock analysis...")
    stock_results = analyze_stocks()
    print("\nComplete Analysis:")
    print(stock_results)

    print("\nTraining machine learning model...")
    ml_model = train_ml_model(stock_results, model_type="RandomForest")
    print("Model training complete.")