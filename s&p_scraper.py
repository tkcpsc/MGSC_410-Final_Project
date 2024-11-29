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
    "AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "BRK.B", "GOOG", "AVGO",
    "JPM", "LLY", "UNH", "XOM", "V", "MA", "COST", "HD", "PG", "WMT", "NFLX", "JNJ",
    "CRM", "BAC", "ABBV", "ORCL", "CVX", "WFC", "MRK", "KO", "CSCO", "ADBE", "ACN",
    "AMD", "PEP", "NOW", "LIN", "MCD", "IBM", "DIS", "PM", "ABT", "GE", "CAT", "TMO",
    "ISRG", "GS", "VZ", "TXN", "INTU", "QCOM", "BKNG", "AXP", "SPGI", "T", "CMCSA",
    "MS", "RTX", "NEE", "PGR", "LOW", "DHR", "AMGN", "UBER", "ETN", "HON", "UNP",
    "PFE", "AMAT", "BLK", "TJX", "COP", "BX", "SYK", "C", "BSX", "PLTR", "PANW",
    "FI", "ADP", "SCHW", "VRTX", "TMUS", "BMY", "DE", "MMC", "SBUX", "GILD", "MU",
    "LMT", "BA", "MDT", "ADI", "KKR", "CB", "PLD", "ANET", "INTC", "UPS", "MO",
    "SO", "AMT", "LRCX", "TT", "CI", "NKE", "ELV", "GEV", "EQIX", "ICE", "SHW",
    "PH", "DUK", "PYPL", "APH", "MDLZ", "CMG", "PNC", "CDNS", "KLAC", "SNPS",
    "AON", "CME", "CRWD", "USB", "CEG", "WM", "MSI", "MCK", "WELL", "REGN", "ZTS",
    "CL", "MCO", "CTAS", "EMR", "EOG", "ITW", "APD", "CVS", "COF", "MMM", "GD",
    "ORLY", "WMB", "CSX", "TDG", "AJG", "ADSK", "FDX", "MAR", "NOC", "OKE", "BDX",
    "CARR", "TFC", "ECL", "NSC", "FCX", "HLT", "SLB", "GM", "ABNB", "FTNT", "HCA",
    "PCAR", "ROP", "TRV", "BK", "DLR", "SRE", "TGT", "FICO", "NXPI", "URI", "RCL",
    "AFL", "AMP", "SPG", "PSX", "JCI", "VST", "CPRT", "PSA", "ALL", "KMI", "GWW",
    "AZO", "AEP", "MPC", "CMI", "MET", "ROST", "PWR", "O", "D", "DHI", "AIG", "NEM",
    "FAST", "HWM", "MSCI", "PEG", "KMB", "PAYX", "LHX", "FIS", "KVUE", "CCI", "PRU",
    "PCG", "DFS", "AME", "TEL", "AXON", "VLO", "RSG", "TRGP", "CTVA", "COR", "F",
    "BKR", "EW", "ODFL", "CBRE", "IR", "VRSK", "LEN", "DAL", "OTIS", "DELL", "HES",
    "IT", "KR", "CTSH", "XEL", "EA", "EXC", "A", "YUM", "MNST", "HPQ", "VMC", "CHTR",
    "GEHC", "ACGL", "SYY", "GLW", "MTB", "KDP", "RMD", "GIS", "MCHP", "LULU", "STZ",
    "NUE", "MLM", "EXR", "IRM", "HIG", "HUM", "WAB", "ED", "DD", "IQV", "IDXX", "NDAQ",
    "VICI", "EIX", "ROK", "OXY", "AVB", "ETR", "FANG", "CSGP", "GRMN", "FITB", "WTW",
    "WEC", "EFX", "EBAY", "UAL", "CNC", "RJF", "DXCM", "DOW", "TTWO", "ANSS", "ON",
    "XYL", "TSCO", "KEYS", "GPN", "CAH", "DECK", "TPL", "STT", "PPG", "HPE", "NVR",
    "DOV", "KHC", "GDDY", "PHM", "HAL", "MPWR", "FTV", "BR", "TROW", "SW", "TYL", "EQT",
    "CHD", "BRO", "AWK", "VLTO", "NTAP", "SYF", "VTR", "CPAY", "HBAN", "EQR", "MTD",
    "DTE", "PPL", "ADM", "CCL", "HSY", "AEE", "RF", "CINF", "HUBB", "SBAC", "PTC", "WDC",
    "CDW", "DVN", "ATO", "IFF", "EXPE", "WY", "WST", "WAT", "BIIB", "CBOE", "ES", "WBD",
    "ZBH", "TDY", "LDOS", "NTRS", "PKG", "K", "LYV", "FE", "BLDR", "CFG", "LYB", "STX",
    "STE", "CNP", "CMS", "NRG", "ZBRA", "CLX", "STLD", "DRI", "FSLR", "IP", "OMC", "COO",
    "LH", "ESS", "CTRA", "MKC", "SNA", "INVH", "WRB", "LUV", "MAA", "BALL", "PODD", "FDS",
    "PFG", "HOLX", "KEY", "TSN", "DGX", "PNR", "LVS", "GPC", "TER", "TRMB", "J", "MAS",
    "IEX", "MOH", "ARE", "BBY", "SMCI", "ULTA", "EXPD", "KIM", "NI", "EL", "BAX", "GEN",
    "EG", "DPZ", "AVY", "DG", "LNT", "ALGN", "TXT", "CF", "L", "DOC", "VTRS", "VRSN",
    "JBHT", "JBL", "AMCR", "EVRG", "APTV", "FFIV", "POOL", "ROL", "MRNA", "RVTY", "EPAM",
    "AKAM", "NDSN", "TPR", "DLTR", "UDR", "SWK", "SWKS", "CPT", "KMX", "CAG", "HST",
    "SJM", "BG", "JKHY", "DAY", "ALB", "CHRW", "EMN", "UHS", "REG", "ALLE", "BXP", "INCY",
    "NCLH", "JNPR", "AIZ", "TECH", "GNRC", "IPG", "PAYC", "NWSA", "LW", "CTLT", "ERIE",
    "TAP", "PNW", "FOXA", "LKQ", "CRL", "GL", "SOLV", "MKTX", "HSIC", "ENPH", "HRL",
    "CPB", "TFX", "RL", "AES", "AOS", "FRT", "MGM", "WYNN", "MTCH", "HAS", "CZR", "APA",
    "IVZ", "MOS", "CE", "BWA", "HII", "DVA", "BF.B", "FMC", "MHK", "BEN", "QRVO", "PARA",
    "WBA", "FOX", "NWS", "AMTM"
]

# SP500_TICKERS = [
#     "AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "BRK.B", "GOOG", "AVGO",
#     "JPM", "LLY", "UNH", "XOM", "V", "MA", "COST", "HD", "PG", "WMT", "NFLX", "JNJ",
#     "CRM", "BAC", "ABBV", "ORCL", "CVX", "WFC", "MRK", "KO", "CSCO", "ADBE", "ACN",
#     "AMD", "PEP", "NOW", "LIN", "MCD", "IBM", "DIS", "PM", "ABT", "GE", "CAT", "TMO"
# ]


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

    for idx, ticker in enumerate(tickers, start=1):
        try:
            # Fetch numerical data for the ticker
            numerical_data = get_ticker_numerical_data(ticker)

            # Skip if ticker data is invalid
            if not numerical_data.get("ticker"):
                print(f"Skipping invalid ticker data for {ticker}")
                continue

            # Fetch Yahoo Finance news links
            news_links = get_yahoo_finance_news_links(ticker, driver)

            # Add up to 14 articles
            for i in range(14):
                if i < len(news_links):
                    article_text = get_article_text_from_url(news_links[i])
                    numerical_data[f"Article {i + 1}"] = article_text if article_text else "no article"
                else:
                    numerical_data[f"Article {i + 1}"] = "no article"

            # Append the complete data
            all_data.append(numerical_data)

        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")

        # Display progress
        update_progress(idx, len(tickers))

    driver.quit()
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

    
    