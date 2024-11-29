import sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import requests
import re

# List of S&P 500 tickers
# SP500_TICKERS = [
#     "TSLA", "NVDA", "JPM", "META"
# ]

# List of S&P 500 tickers
SP500_TICKERS = [
    "NVDA", "JPM", 
    "HD", "PYPL", "BAC"
]


# Configure Selenium WebDriver
def configure_webdriver():
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--blink-settings=imagesEnabled=false")  # Disable loading images
    service = Service()
    driver = webdriver.Chrome(service=service, options=options)
    return driver

# Clean article text
def clean_article_text(text):
    if text:
        # Remove punctuation and extra whitespace
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
        return text
    return None

# Get Yahoo Finance news links using Selenium
def get_yahoo_finance_news_links(ticker, driver):
    url = f'https://finance.yahoo.com/quote/{ticker}'
    try:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        news_items = soup.find_all('section', {'data-testid': 'storyitem'})

        links = []
        for item in news_items:
            link = item.find('a', href=True)
            if link and link['href'].startswith('http'):
                links.append(link['href'])
        return links[:14]  # Limit to 14 news links
    except Exception as e:
        print(f"\nError fetching news links for {ticker}: {e}")
        return []

# Fetch article text using requests and BeautifulSoup
def get_article_text_from_url(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, 'html.parser')
        article_body = soup.find('div', class_='article-wrap') or soup.find('div', class_='caas-body') or soup.find('div', class_='body')
        return clean_article_text(article_body.get_text(separator=' ', strip=True)) if article_body else "Could not retreive article"
    except Exception:
        return None


import yfinance as yf
from datetime import datetime

# Fetch stock data using yfinance
def get_ticker_numerical_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        stock_data = {
            "ticker": ticker,
            "date_and_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Previous Close": info.get("previousClose", "N/A"),
            "Open": info.get("open", "N/A"),
            "Bid": info.get("bid", "N/A"),
            "Ask": info.get("ask", "N/A"),
            "Day's Range": f"{info.get('dayLow', 'N/A')} - {info.get('dayHigh', 'N/A')}",
            "52 Week Range": f"{info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}",
            "Volume": info.get("volume", "N/A"),
            "Avg. Volume": info.get("averageVolume", "N/A"),
            "Market Cap (intraday)": info.get("marketCap", "N/A"),
            "Beta (5Y Monthly)": info.get("beta", "N/A"),
            "PE Ratio (TTM)": info.get("trailingPE", "N/A"),
            "EPS (TTM)": info.get("trailingEps", "N/A"),
            "1y Target Est": info.get("targetMeanPrice", "N/A")
        }

        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return {}

# Display progress
def update_progress(current, total):
    progress = (current / total) * 100
    sys.stdout.write(f"\rProgress: {progress:.2f}% complete")
    sys.stdout.flush()

def fetch_data_for_all_tickers(tickers):
    driver = configure_webdriver()
    all_data = []
    total_tasks = len(tickers)

    for idx, ticker in enumerate(tickers, start=1):
        try:
            # Check if the last row in all_data has a valid ticker
            if all_data and not isinstance(all_data[-1].get("ticker"), str):
                print(f"Deleting invalid row: {all_data[-1]}")
                all_data.pop()  # Remove the last invalid row

            # Fetch numerical data for the ticker
            numerical_data = get_ticker_numerical_data(ticker)
            news_links = get_yahoo_finance_news_links(ticker, driver)

            # Fetch and assign articles
            for i in range(14):  # Ensure exactly 14 article slots
                if i < len(news_links):
                    url = news_links[i]
                    article_text = get_article_text_from_url(url)
                    numerical_data[f"Article {i + 1}"] = article_text if article_text else "no article"
                else:
                    # Fill missing articles with "no article"
                    numerical_data[f"Article {i + 1}"] = "no article"

            # Append the processed data
            all_data.append(numerical_data)
        except Exception as e:
            print(f"\nError processing ticker {ticker}: {e}")
        finally:
            update_progress(idx, total_tasks)

    driver.quit()
    sys.stdout.write("\n")  # Newline after progress completes
    return all_data

# Export data to CSV
def export_to_csv(all_stock_data, filename='sp500_stock_data.csv'):
    print(f"Exporting data to CSV: {filename}")
    df = pd.json_normalize(all_stock_data)
    df.to_csv(filename, index=False)
    print(f"Exported data to {filename}")

# Entry point
if __name__ == "__main__":
    print("Starting data fetch...")
    all_stock_data = fetch_data_for_all_tickers(SP500_TICKERS)
    export_to_csv(all_stock_data)

    
    