import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

# Example CSV file paths
input_file_path = "data/merged.csv"  # Replace with your input file path
output_file_path = "data/processed.csv"

# Load the CSV
data = pd.read_csv(input_file_path)

# Function to fetch the closing price, tracking how many days forward/back it needs to go
def fetch_closing_price(ticker, target_date, max_retries=7, direction="backward"):
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

# Add new columns for t, t-1 to t-5, and t+1
new_columns = ["t", "t+1"] + [f"t-{i}" for i in range(1, 6)]
for col in new_columns:
    data[col] = None

# Iterate over each row in the input CSV
for index, row in data.iterrows():
    first_row_date_str = row['date_and_time']
    ticker = row['ticker']

    # Parse the date
    try:
        parsed_date = datetime.strptime(first_row_date_str, "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        print(f"Error parsing date for row {index}: {e}")
        continue

    backtrack_days = 0  # Track how many days were backtracked from parsed_date

    # Fetch t (current date)
    t_price, t_days_back = fetch_closing_price(ticker, parsed_date)
    if t_price is not None:
        data.at[index, "t"] = t_price
        print(f"Row {index}: t on {parsed_date.date()} Closing Price: {t_price} (Backtracked {t_days_back} days)")

    # Fetch t-1 to t-5
    for t in range(1, 6):  # t-1 through t-5
        target_date = parsed_date - timedelta(days=backtrack_days + 1)
        price, days_back = fetch_closing_price(ticker, target_date)
        if price is not None:
            effective_date = target_date - timedelta(days=days_back)
            data.at[index, f"t-{t}"] = price
            backtrack_days += days_back + 1
            print(f"Row {index}: t-{t} on {effective_date} Closing Price: {price} (Backtracked {days_back} days)")

    # Fetch t+1 (next valid trading day)
    target_date = parsed_date + timedelta(days=1)
    t_plus_1_price, t_plus_1_days_forward = fetch_closing_price(ticker, target_date, direction="forward")
    if t_plus_1_price is not None:
        effective_date = target_date + timedelta(days=t_plus_1_days_forward)
        data.at[index, "t+1"] = t_plus_1_price
        print(f"Row {index}: t+1 on {effective_date} Closing Price: {t_plus_1_price} (Forwarded {t_plus_1_days_forward} days)")

# Save the updated DataFrame to a new CSV
data.to_csv(output_file_path, index=False)

print(f"Processed data with all original and new columns has been saved to {output_file_path}.")