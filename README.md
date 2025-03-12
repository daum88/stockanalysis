# Stock Analysis and Prediction using Machine Learning

## Overview
This project analyzes stock market data using financial indicators, sentiment analysis, and machine learning models. It fetches stock data from Yahoo Finance, computes technical indicators, retrieves sentiment scores from financial news, and applies machine learning models to predict stock prices.

## Features
- Fetch stock data from **Yahoo Finance**
- Compute key **technical indicators** (RSI, MACD, SMA)
- Perform **news sentiment analysis** using VADER Sentiment Analyzer
- Evaluate **stock fair value** using a simplified Discounted Cash Flow (DCF) model
- Train and test **machine learning models** (Random Forest & Neural Networks) to predict stock prices

## Requirements
To use this project, install the required dependencies using:
```bash
pip install -r requirements.txt
```

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/daum88/stockanalysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd stockanalysis
   ```
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the analysis:
   ```bash
   python main.py
   ```

## Machine Learning Models
- **Random Forest Regressor**
- **MLP Neural Network**
- Features used: P/E Ratio, EPS, Market Cap, 52-Week High/Low, Beta, Dividend Yield, Revenue Growth, Sentiment Score, RSI, MACD, SMA
