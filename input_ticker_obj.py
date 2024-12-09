import sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import requests
import re
import yfinance as yf
import openai
import json
import numpy as np
import os
from dotenv import load_dotenv
import csv


class StockData:
    def __init__(self, ticker):
    
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

        # Fetch data for the ticker
        all_stock_data = fetch_data_for_single_ticker(ticker)
        if all_stock_data:
            # Create a DataFrame from the fetched data
            self.df = pd.DataFrame(all_stock_data)
        else:
            # Assign an empty DataFrame if no data is available
            self.df = pd.DataFrame()        
        # Initialize variables for each article (1 to 14) with null values
        for article_num in range(1, 15):
            self.df[f'sentiment_score_article_{article_num}'] = np.nan
            self.df[f'ticker_mentions_article_{article_num}'] = np.nan
            self.df[f'competitor_mentions_article_{article_num}'] = np.nan
            self.df[f'urgence_indicators_article_{article_num}'] = None
            self.df[f'event_detection_article_{article_num}'] = None
            self.df[f'length_article_{article_num}'] = np.nan
            self.df[f'complexity_article_{article_num}'] = None
            self.df[f'relevance_article_{article_num}'] = None

        # Example to verify initialization
        print("\n\nEVERYTHING SHOULD BE NULL FOR LLM\n", self.df.head())
        
        # Load environment variables from key.env
        load_dotenv("key.env")

        # Access the API key
        openai.api_key = os.getenv("OPENAI_API_KEY")

        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        print(f"Loaded OpenAI API key: {openai.api_key[:6]}...")  # For debugging (hides most of the key)
           
            
        prompt = (
            f"""
            You are an advanced financial text analysis assistant. 
            Your task is to analyze a stock-related dataframe containing articles and add the following features. 
            ONLY return a list of the following features for each article using articles 1-14:  
            sentiment_score_article_1 (a value between -1 for negative, 0 for neutral, and 1 for positive), 
            ticker_mentions_article_1 (Number of times the company or ticker symbol is mentioned), 
            competitor_mentions_article_1 (Number of competitor mentions), 
            urgency_indicators_article_1 (Whether the language in the article suggests urgency (Yes/No)), 
            event_detection_article_1 (Does the article mention significant events (e.g., earnings, merger, 
            product launch)? (Yes/No)), length_article_1 (Article Length (word count)), 
            complexity_article_1 (Complexity of language (e.g., readability score, or "simple"/"complex")), 
            relevance_article_1 (Relevance to the company (high, medium, low))  
            if there is no article, input the corresponding null value for that specific features type (0, null, "none", etc.).
            Only return a list of each new feature per article. do not return any code or anything besides a dictionary of the feature name and the coresponding value.
            \n\narticle 1\n,{self.get_article(1)}
            \n\narticle 2\n,{self.get_article(2)}
            \n\narticle 3\n,{self.get_article(3)}
            \n\narticle 4\n,{self.get_article(4)}
            \n\narticle 5\n,{self.get_article(5)}
            \n\narticle 6\n,{self.get_article(6)}
            \n\narticle 7\n,{self.get_article(7)}
            \n\narticle 8\n,{self.get_article(8)}
            \n\narticle 9\n,{self.get_article(9)}
            \n\narticle 10\n,{self.get_article(10)}
            \n\narticle 11\n,{self.get_article(11)}
            \n\narticle 12\n,{self.get_article(12)}
            \n\narticle 13\n,{self.get_article(13)}
            \n\narticle 14\n,{self.get_article(14)}

        """
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,  # Try to lower this later
                temperature=0.5
            )

            strategy = response.choices[0].message['content'].strip()

            # Remove backticks and JSON label
            if strategy.startswith("```json"):
                strategy = strategy[7:-3].strip()

            # Wrap the response in brackets if it contains multiple top-level JSON objects
            if not strategy.startswith("["):
                strategy = f"[{strategy}]"

            # Validate and parse the strategy JSON
            if not strategy:
                print("LLM response is empty.")
                return None

            try:
                parsed_strategy = json.loads(strategy)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"LLM response: {strategy}")
                return None

            # Ensure the DataFrame is already initialized and has the required columns
            if self.df.empty:
                print("The DataFrame is empty. Ensure data initialization is done before processing.")
                return None

            # Iterate over each article in the parsed strategy and update the DataFrame
            for i, article_data in enumerate(parsed_strategy, start=1):
                for key, value in article_data.items():
                    column_name = f"{key}"
                    if column_name in self.df.columns:
                        self.df.at[0, column_name] = value  # Update the first row with the value

            print("\n\nEVERYTHING SHOULD BE POPULATED FOR LLM\n", self.df.head())

        except openai.OpenAIError as e:
            print(f"OpenAI API error: {e}")
            return None

        
    def view_columns(self):
        """
        View the columns of the DataFrame.
        :return: List of column names or a message if the DataFrame is empty.
        """
        if self.df is not None and not self.df.empty:
            return self.df.columns.tolist()
        else:
            return "The DataFrame is empty or not initialized."    

    def get_ticker(self):
        return self.df['ticker'].iloc[0] if 'ticker' in self.df.columns else None

    def get_date_and_time(self):
        return self.df['date_and_time'].iloc[0] if 'date_and_time' in self.df.columns else None

    def get_previous_close(self):
        return self.df['Previous Close'].iloc[0] if 'Previous Close' in self.df.columns else None

    def get_open(self):
        return self.df['Open'].iloc[0] if 'Open' in self.df.columns else None

    def get_bid(self):
        return self.df['Bid'].iloc[0] if 'Bid' in self.df.columns else None

    def get_ask(self):
        return self.df['Ask'].iloc[0] if 'Ask' in self.df.columns else None

    def get_days_range(self):
        return self.df["Day's Range"].iloc[0] if "Day's Range" in self.df.columns else None

    def get_52_week_range(self):
        return self.df['52 Week Range'].iloc[0] if '52 Week Range' in self.df.columns else None

    def get_volume(self):
        return self.df['Volume'].iloc[0] if 'Volume' in self.df.columns else None

    def get_avg_volume(self):
        return self.df['Avg. Volume'].iloc[0] if 'Avg. Volume' in self.df.columns else None

    def get_market_cap_intraday(self):
        return self.df['Market Cap (intraday)'].iloc[0] if 'Market Cap (intraday)' in self.df.columns else None

    def get_beta_5y_monthly(self):
        return self.df['Beta (5Y Monthly)'].iloc[0] if 'Beta (5Y Monthly)' in self.df.columns else None

    def get_pe_ratio_ttm(self):
        return self.df['PE Ratio (TTM)'].iloc[0] if 'PE Ratio (TTM)' in self.df.columns else None

    def get_eps_ttm(self):
        return self.df['EPS (TTM)'].iloc[0] if 'EPS (TTM)' in self.df.columns else None

    def get_1y_target_est(self):
        return self.df['1y Target Est'].iloc[0] if '1y Target Est' in self.df.columns else None

    # Article accessors
    def get_article(self, article_number):
        column_name = f"Article {article_number}"
        return self.df[column_name].iloc[0] if column_name in self.df.columns else None
    
    
    def get_article_with_attributes(self, article_number):
        """
        Get the LLM attributes and the text of a specific article from the DataFrame.
        :param article_number: The article number to view (1-14).
        :return: A string with attributes and article text, or a message if the article is not found.
        """
        attributes = [
            f"sentiment_score_article_{article_number}",
            f"ticker_mentions_article_{article_number}",
            f"competitor_mentions_article_{article_number}",
            f"urgence_indicators_article_{article_number}",
            f"event_detection_article_{article_number}",
            f"length_article_{article_number}",
            f"complexity_article_{article_number}",
            f"relevance_article_{article_number}",
        ]
        
        if self.df is not None and not self.df.empty:
            result = []
            for attr in attributes:
                value = self.df[attr].iloc[0] if attr in self.df.columns else None
                result.append(f"{attr}: {value}")
            
            # Add a blank line between attributes and article text
            column_name = f"Article {article_number}"
            article_text = self.df[column_name].iloc[0] if column_name in self.df.columns else "No article text found"
            result.append("")  # Add an empty string for a blank line
            result.append(f"Article Text:\n{article_text}")
            
            return "\n".join(result)
        else:
            return "The DataFrame is empty or not initialized."#     """

    def fetch_numerical_data(self):
        """
        Fetches numerical stock data using yfinance.
        """
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info

            # Populate numerical data
            self.numerical_data = {
                "ticker": self.ticker,
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

            # Convert to DataFrame for export
            self.df = pd.DataFrame([self.numerical_data])

        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {e}")
            self.numerical_data = {}
            self.df = pd.DataFrame()

    def fetch_article_attributes(self):
            """
            Fetches the specified attributes from self.df.head() in order.

            Returns:
                dict: A dictionary containing the specified attributes with their values.
            """
            attributes = [
                "sentiment_score_article_1", "ticker_mentions_article_1", "competitor_mentions_article_1", "urgency_indicators_article_1", "event_detection_article_1", "length_article_1", "complexity_article_1", "relevance_article_1",
                "sentiment_score_article_2", "ticker_mentions_article_2", "competitor_mentions_article_2", "urgency_indicators_article_2", "event_detection_article_2", "length_article_2", "complexity_article_2", "relevance_article_2",
                "sentiment_score_article_3", "ticker_mentions_article_3", "competitor_mentions_article_3", "urgency_indicators_article_3", "event_detection_article_3", "length_article_3", "complexity_article_3", "relevance_article_3",
                "sentiment_score_article_4", "ticker_mentions_article_4", "competitor_mentions_article_4", "urgency_indicators_article_4", "event_detection_article_4", "length_article_4", "complexity_article_4", "relevance_article_4",
                "sentiment_score_article_5", "ticker_mentions_article_5", "competitor_mentions_article_5", "urgency_indicators_article_5", "event_detection_article_5", "length_article_5", "complexity_article_5", "relevance_article_5",
                "sentiment_score_article_6", "ticker_mentions_article_6", "competitor_mentions_article_6", "urgency_indicators_article_6", "event_detection_article_6", "length_article_6", "complexity_article_6", "relevance_article_6",
                "sentiment_score_article_7", "ticker_mentions_article_7", "competitor_mentions_article_7", "urgency_indicators_article_7", "event_detection_article_7", "length_article_7", "complexity_article_7", "relevance_article_7",
                "sentiment_score_article_8", "ticker_mentions_article_8", "competitor_mentions_article_8", "urgency_indicators_article_8", "event_detection_article_8", "length_article_8", "complexity_article_8", "relevance_article_8",
                "sentiment_score_article_9", "ticker_mentions_article_9", "competitor_mentions_article_9", "urgency_indicators_article_9", "event_detection_article_9", "length_article_9", "complexity_article_9", "relevance_article_9",
                "sentiment_score_article_10", "ticker_mentions_article_10", "competitor_mentions_article_10", "urgency_indicators_article_10", "event_detection_article_10", "length_article_10", "complexity_article_10", "relevance_article_10",
                "sentiment_score_article_11", "ticker_mentions_article_11", "competitor_mentions_article_11", "urgency_indicators_article_11", "event_detection_article_11", "length_article_11", "complexity_article_11", "relevance_article_11",
                "sentiment_score_article_12", "ticker_mentions_article_12", "competitor_mentions_article_12", "urgency_indicators_article_12", "event_detection_article_12", "length_article_12", "complexity_article_12", "relevance_article_12",
                "sentiment_score_article_13", "ticker_mentions_article_13", "competitor_mentions_article_13", "urgency_indicators_article_13", "event_detection_article_13", "length_article_13", "complexity_article_13", "relevance_article_13",
                "sentiment_score_article_14", "ticker_mentions_article_14", "competitor_mentions_article_14", "urgency_indicators_article_14", "event_detection_article_14", "length_article_14", "complexity_article_14", "relevance_article_14"
            ]
            # Fetch the first row from self.df
            if self.df is not None and not self.df.empty:
                first_row = self.df.head(1).to_dict(orient="records")[0]
                return {attr: first_row.get(attr, "N/A") for attr in attributes}
            else:
                return {attr: "N/A" for attr in attributes}
            
            


    def fetch_t_values(self):
        """
        Fetches closing prices for t, t+1, and t-1 through t-5 for the ticker in self.df.
        Returns a dictionary with the values in the correct order.
        """
        # Initialize the t_values dictionary in the correct order
        t_values = {"t": None, "t+1": None, "t-1": None, "t-2": None, "t-3": None, "t-4": None, "t-5": None}

        # Ensure the DataFrame contains the required data
        if self.df is None or self.df.empty:
            print("DataFrame is empty. Cannot fetch t values.")
            return t_values

        try:
            # Retrieve ticker and date
            ticker = self.df.iloc[0]['ticker']
            first_row_date_str = self.df.iloc[0]['date_and_time']
            parsed_date = datetime.strptime(first_row_date_str, "%Y-%m-%d %H:%M:%S")
        except KeyError as e:
            print(f"Missing column: {e}")
            return t_values
        except ValueError as e:
            print(f"Invalid date format: {e}")
            return t_values

        def fetch_closing_price(ticker, target_date, max_retries=7, direction="backward"):
            """
            Fetches the closing price for a given ticker and date, with retries.
            """
            retry_count = 0
            delta = -1 if direction == "backward" else 1
            while retry_count < max_retries:
                try:
                    stock = yf.Ticker(ticker)
                    start_date = target_date.strftime('%Y-%m-%d')
                    end_date = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
                    history = stock.history(start=start_date, end=end_date)
                    if not history.empty:
                        closing_price = history['Close'].iloc[0]
                        return closing_price, retry_count
                    else:
                        target_date += timedelta(days=delta)
                        retry_count += 1
                except Exception as e:
                    print(f"Error fetching data for {ticker} on {target_date.date()}: {e}")
                    target_date += timedelta(days=delta)
                    retry_count += 1
            return None, retry_count

        # Initialize backtracking
        backtrack_days = 0

        # Fetch t (current date)
        t_price, t_days_back = fetch_closing_price(ticker, parsed_date)
        if t_price is not None:
            t_values["t"] = t_price
            print(f"t on {parsed_date.date()} Closing Price: {t_price} (Backtracked {t_days_back} days)")

        # Fetch t-1 to t-5
        for i in range(1, 6):
            target_date = parsed_date - timedelta(days=backtrack_days + 1)
            price, days_back = fetch_closing_price(ticker, target_date)
            if price is not None:
                effective_date = target_date - timedelta(days=days_back)
                t_values[f"t-{i}"] = price
                backtrack_days += days_back + 1
                print(f"t-{i} on {effective_date.date()} Closing Price: {price} (Backtracked {days_back} days)")

        # Fetch t+1
        target_date = parsed_date + timedelta(days=1)
        t_plus_1_price, t_plus_1_days_forward = fetch_closing_price(ticker, target_date, direction="forward")
        if t_plus_1_price is not None:
            effective_date = target_date + timedelta(days=t_plus_1_days_forward)
            t_values["t+1"] = t_plus_1_price
            print(f"t+1 on {effective_date.date()} Closing Price: {t_plus_1_price} (Forwarded {t_plus_1_days_forward} days)")

        return t_values


        
    def export_to_csv(self, filename="data/user_testing_data.csv"):
        """
        Exports the stock data to a CSV, including t-values and dummy variables for categorical columns.
        """
        required_columns = [
            "ticker", "date_and_time", "Previous Close", "Open", "Bid", "Ask",
            "Day's Range", "52 Week Range", "Volume", "Avg. Volume",
            "Market Cap (intraday)", "Beta (5Y Monthly)", "PE Ratio (TTM)",
            "EPS (TTM)", "1y Target Est"
        ]

        if self.df is not None and not self.df.empty:
            # Ensure the DataFrame contains all required columns
            for column in required_columns:
                if column not in self.df.columns:
                    self.df[column] = "N/A"

            # Fetch article attributes and t-values
            article_attributes = self.fetch_article_attributes()
            t_values = self.fetch_t_values()

            # Add article attributes and t-values to the DataFrame
            for attr, value in {**article_attributes, **t_values}.items():
                if attr not in self.df.columns:
                    self.df[attr] = "N/A"
                self.df.at[0, attr] = value

            # Reorder columns
            all_columns = required_columns + list(article_attributes.keys()) + list(t_values.keys())
            for column in all_columns:
                if column not in self.df.columns:
                    self.df[column] = "N/A"
            self.df = self.df[all_columns]

            try:
                self.df.to_csv(filename, index=False, quoting=csv.QUOTE_MINIMAL)
                print(f"Data exported successfully to {filename}.")
            except Exception as e:
                print(f"Failed to export data to CSV: {e}")
        else:
            print("No data available to export.")

        import pandas as pd
        import numpy as np

        # Load the CSV file into a pandas DataFrame
        data = pd.read_csv(filename)

        # Fill missing values in sentiment_score_article_1 to sentiment_score_article_14 with 0
        sentiment_columns = [f'sentiment_score_article_{i}' for i in range(1, 15)]
        data[sentiment_columns] = data[sentiment_columns].fillna(0)

        # Drop 'ticker' and 'date_and_time' columns
        data = data.drop(columns=['ticker', 'date_and_time'], errors='ignore')

        # Convert boolean columns to integers (True -> 1, False -> 0)
        bool_columns = data.select_dtypes(include=['bool']).columns
        data[bool_columns] = data[bool_columns].astype(int)

        # Define a function to split range columns
        def split_range_column(df, column):
            """Splits a range column like 'low - high' into two separate numeric columns."""
            ranges = df[column].str.split(' - ', expand=True)
            df[f'{column}_min'] = pd.to_numeric(ranges[0], errors='coerce')
            df[f'{column}_max'] = pd.to_numeric(ranges[1], errors='coerce')
            df.drop(columns=[column], inplace=True)

        # Process range columns
        if "Day's Range" in data.columns:
            split_range_column(data, "Day's Range")
        if "52 Week Range" in data.columns:
            split_range_column(data, "52 Week Range")

        # Drop categorical columns directly
        categorical_columns = [
            "urgency_indicators_article_1", "event_detection_article_1", "complexity_article_1", "relevance_article_1",
            "urgency_indicators_article_2", "event_detection_article_2", "complexity_article_2", "relevance_article_2",
            "urgency_indicators_article_3", "event_detection_article_3", "complexity_article_3", "relevance_article_3",
            "urgency_indicators_article_4", "event_detection_article_4", "complexity_article_4", "relevance_article_4",
            "urgency_indicators_article_5", "event_detection_article_5", "complexity_article_5", "relevance_article_5",
            "urgency_indicators_article_6", "event_detection_article_6", "complexity_article_6", "relevance_article_6",
            "urgency_indicators_article_7", "event_detection_article_7", "complexity_article_7", "relevance_article_7",
            "urgency_indicators_article_8", "event_detection_article_8", "complexity_article_8", "relevance_article_8",
            "urgency_indicators_article_9", "event_detection_article_9", "complexity_article_9", "relevance_article_9",
            "urgency_indicators_article_10", "event_detection_article_10", "complexity_article_10", "relevance_article_10",
            "urgency_indicators_article_11", "event_detection_article_11", "complexity_article_11", "relevance_article_11",
            "urgency_indicators_article_12", "event_detection_article_12", "complexity_article_12", "relevance_article_12",
            "urgency_indicators_article_13", "event_detection_article_13", "complexity_article_13", "relevance_article_13",
            "urgency_indicators_article_14", "event_detection_article_14", "complexity_article_14", "relevance_article_14"
        ]
        data = data.drop(columns=categorical_columns, errors='ignore')

        # Save the prepared dataset to a new CSV file
        prepared_file_path = 'data/user_input_cleaned.csv'
        data.to_csv(prepared_file_path, index=False)

        print(f"Prepared dataset saved to {prepared_file_path}")        
    
    
    
    
stock_data = StockData("NVDA")
stock_data.export_to_csv()  








#     View the news links stored in the object.
#     :return: List of news links or a message if the list is empty.
#     """
#     if self.news_links is not None and len(self.news_links) > 0:
#         return self.news_links
#     else:
#         return "There are no news links."




# Create an instance of the class
# stock_data = StockData("NVDA")

# View the columns of the DataFrame
# print(stock_data.view_columns())
# print(stock_data.view_news_links())

# Access different parts of the DataFrame
# print(stock_data.get_ticker())  # Outputs the ticker
# print(stock_data.get_previous_close())  # Outputs the previous close value
# print("\n\narticle 1\n", stock_data.get_article(1))  # Outputs the text for "Article 1"
# print("\n\narticle 2\n", stock_data.get_article(2))  # Outputs the text for "Article 1"

# print("\n\n\n LLM: \n\n")
# stock_data.llm_hit()
