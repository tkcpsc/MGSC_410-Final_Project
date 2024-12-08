import sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import requests
import re
import yfinance as yf


# Configure Selenium WebDriver
def configure_webdriver():
    print("Entering function: configure_webdriver")
    try:
        print("Configuring Selenium WebDriver...")
        options = Options()
        options.add_argument("--headless")  # Run in headless mode
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--blink-settings=imagesEnabled=false")  # Disable loading images
        service = Service()
        driver = webdriver.Chrome(service=service, options=options)
        print("WebDriver configured successfully.")
        return driver
    except Exception as e:
        print(f"Error initializing Selenium WebDriver: {e}")
        sys.exit(1)


# Clean article text
def clean_article_text(text):
    print("Entering function: clean_article_text")
    print(f"Original text: {text[:100]}...")  # Display first 100 characters
    if text:
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
        print(f"Cleaned text: {text[:100]}...")  # Display first 100 characters
        return text
    print("No text to clean.")
    return None


# Get Yahoo Finance news links using Selenium
def get_yahoo_finance_news_links(ticker, driver):
    print(f"Entering function: get_yahoo_finance_news_links for ticker: {ticker}")
    url = f'https://finance.yahoo.com/quote/{ticker}'
    print(f"Navigating to URL: {url}")
    try:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        news_items = soup.find_all('section', {'data-testid': 'storyitem'})

        links = []
        print("Extracting links from news items...")
        for item in news_items:
            link = item.find('a', href=True)
            if link and link['href'].startswith('http'):
                print(f"Found link: {link['href']}")
                links.append(link['href'])
        print(f"Total links extracted: {len(links)}")
        return links[:14]  # Limit to 14 news links
    except Exception as e:
        print(f"\nError fetching news links for {ticker}: {e}")
        return []


# Fetch article text using requests and BeautifulSoup
def get_article_text_from_url(url):
    print(f"Entering function: get_article_text_from_url for URL: {url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        print("Sending HTTP GET request...")
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Response status code: {response.status_code}")
        if response.status_code != 200:
            print(f"Failed to retrieve article, status code: {response.status_code}")
            return "Could not retrieve article"

        soup = BeautifulSoup(response.content, 'html.parser')
        print("Parsing article content...")
        article_body = soup.find('div', class_='article-wrap') or soup.find('div', class_='caas-body') or soup.find('div', class_='body')
        if article_body:
            print("Article body found.")
            text = clean_article_text(article_body.get_text(separator=' ', strip=True))
        else:
            print("Article body not found.")
            text = "Could not retrieve article"
        print("Article text processed successfully.")
        return text
    except Exception as e:
        print(f"Error fetching article: {e}")
        return "Could not retrieve article"


# Fetch stock data using yfinance
def get_ticker_numerical_data(ticker):
    print(f"Entering function: get_ticker_numerical_data for ticker: {ticker}")
    try:
        print("Fetching data from yfinance...")
        stock = yf.Ticker(ticker)
        info = stock.info
        print("Processing numerical data...")
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
        print("Numerical data fetched successfully.")
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return {}


# Fetch data for a single ticker
def fetch_data_for_single_ticker(ticker):
    print(f"Entering function: fetch_data_for_single_ticker for ticker: {ticker}")
    driver = configure_webdriver()

    try:
        # Fetch numerical data for the ticker
        numerical_data = get_ticker_numerical_data(ticker)

        # Skip if ticker data is invalid
        if not numerical_data.get("ticker"):
            print(f"Invalid ticker: {ticker}")
            return []

        # Fetch Yahoo Finance news links
        news_links = get_yahoo_finance_news_links(ticker, driver)

        # Add up to 14 articles
        print(f"Processing up to 14 articles for ticker: {ticker}")
        for i in range(14):
            if i < len(news_links):
                print(f"Fetching article {i + 1}...")
                article_text = get_article_text_from_url(news_links[i])
                numerical_data[f"Article {i + 1}"] = article_text if article_text else "no article"
            else:
                print(f"No article {i + 1} found.")
                numerical_data[f"Article {i + 1}"] = "no article"

        print(f"Data fetch completed for ticker: {ticker}")
        return [numerical_data]

    except Exception as e:
        print(f"Error processing ticker {ticker}: {e}")
        return []

    finally:
        print("Quitting WebDriver.")
        driver.quit()


# Entry point
if __name__ == "__main__":
    ticker = "AAPL"  # Hardcoded ticker
    print(f"Fetching data for ticker: {ticker}")
    all_stock_data = fetch_data_for_single_ticker(ticker)

    if all_stock_data:
        print(f"Displaying data for ticker: {ticker}")
        for data in all_stock_data:
            for key, value in data.items():
                print(f"{key}: {value}")
    else:
        print(f"No data available for ticker: {ticker}")
