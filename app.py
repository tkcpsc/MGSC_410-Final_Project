from shiny import App, ui, render, reactive
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
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



# Color Palette
PRIMARY_COLOR = "#007BFF"  # Primary color (e.g., blue for headers)
SECONDARY_COLOR = "#D3D3D3"  # Secondary color (e.g., for pie chart empty parts)
WHITE_COLOR = "#ffffff"  # Secondary color (e.g., for pie chart empty parts)
BACKGROUND_COLOR = "#2e2e2e"  # Background color for cards and figure
ACCENT_COLOR = "#FFD700"  # Accent color (e.g., for hover or additional highlights)
BODY_BACKGROUND = "#212121"  # Background color for the entire page

# User Interface
app_ui = ui.page_fluid(
    # Single Card
    ui.card(
        # Card Title
        ui.card_header(
            ui.h2("AI-Powered News Analysis for Stock Market Forecasting", style=f"color: white; margin: 0; text-align: center;"),
            style=f"background-color: {PRIMARY_COLOR}; padding: 10px;",  # Dynamic primary color
        ),
        # Card Content
        ui.div(
            ui.output_ui("circle_output"),  # Placeholder for the chart
            ui.div(
                ui.output_text("percentage_label"),
                style="text-align: center; margin-top: 0px;",  # Reduced margin-top for label
            ),
            # Input text box and generate button
            ui.div(
                ui.div(
                    ui.input_text("input_text", "S&P 500 Stock Ticker:", value=""),
                    style=f"color: {WHITE_COLOR}; margin-top: 10px; width: 80%; text-align: center;",
                ),
                ui.div(
                    ui.input_action_button("generate_btn", "Generate"),
                    style=f"background-color: {PRIMARY_COLOR}; color: white; margin-top: 10px; width: 50%; text-align: center; border-",
                ),
                style="display: flex; flex-direction: column; align-items: center; margin-top: 10px;",  # Center inputs
            ),
            # Add output for stock data
            ui.div(
                ui.output_ui("stock_data_output"),  # Placeholder for fetched stock data
                style="margin-top: 20px; color: white;",  # Add some spacing and color
            ),
            style="display: flex; flex-direction: column; align-items: center; padding: 10px;",  # Reduced padding
        ),
        style=(
            f"background-color: {BACKGROUND_COLOR}; "
            "border-radius: 10px; "
            "padding: 0px; "  # Reduced padding
            "max-width: 700px; "  # Set max width
            "margin: 0 auto;"  # Center the card
        ),
    ),
    # Add CSS for responsiveness and color palette
    ui.tags.style(f"""
        body {{
            background-color: {BODY_BACKGROUND};  /* Set the body background color */
            color: white;  /* Default text color for dark background */
        }}
        #circle_output img {{
            width: 50vw;  /* 50% of the viewport width */
            height: auto; /* Maintain aspect ratio */
            max-width: 500px; /* Limit the maximum size */
        }}
        #percentage_label {{
            font-size: 1.5vw; /* Relative font size to viewport width */
            max-font-size: 20px; /* Limit the maximum font size */
            color: {PRIMARY_COLOR}; /* Primary color for the label */
        }}
    """),
)


# =================================================================

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

# =================================================================






# Server logic
def server(input, output, session):
    # Reactive value to hold the fetched stock data
    stock_data = reactive.Value({})  # Initializes as an empty dictionary

    # Reactive function to fetch data when the "Generate" button is clicked
    @reactive.Effect
    @reactive.event(input.generate_btn)
    def fetch_data():
        ticker = input.input_text()  # Get the ticker from the text input box
        if ticker.strip():
            print(f"Fetching data for ticker: {ticker}")
            fetched_data = fetch_data_for_single_ticker(ticker.strip().upper())
            if fetched_data:
                stock_data.set(fetched_data[0])  # Update the reactive value with the first result
            else:
                stock_data.set({"Error": "No data available for the provided ticker."})
        else:
            stock_data.set({"Error": "Please enter a valid ticker symbol."})

    # Render the fetched stock data below the Generate button
    @output
    @render.ui
    def stock_data_output():
        data = stock_data.get()  # Get the latest stock data
        if not data:
            return ui.HTML("<p>No data to display. Enter a ticker and click Generate.</p>")

        # Format the fetched data as HTML
        formatted_data = "<br>".join(
            f"<b>{key}:</b> {value}" for key, value in data.items()
        )
        return ui.HTML(f"<div style='color: white; text-align: left; padding: 10px;'>{formatted_data}</div>")

    # Reactive value to hold the percentage for the pie chart (example logic)
    percentage = reactive.Value(50)

    # Render the pie chart (example chart logic)
    @output
    @render.ui
    def circle_output():
        figure_width = 6
        fig, ax = plt.subplots(figsize=(figure_width, figure_width), subplot_kw={'aspect': 'equal'})
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.set_facecolor(BACKGROUND_COLOR)
        wedges, _ = ax.pie(
            [percentage.get(), 100 - percentage.get()],
            startangle=90,
            colors=[PRIMARY_COLOR, SECONDARY_COLOR],
            wedgeprops={'edgecolor': 'white'},
        )
        fig.tight_layout()
        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        plt.close(fig)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        img_src = f"data:image/png;base64,{img_base64}"
        return ui.HTML(f'<img src="{img_src}" id="circle_output">')

    # Render the percentage label (example chart logic)
    @output
    @render.text
    def percentage_label():
        return f"Selected Percentage: {percentage.get()}%"

# Create the Shiny app
app = App(app_ui, server)