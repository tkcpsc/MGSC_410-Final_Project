import pandas as pd
import numpy as np

# Define the file path
file_path = 'data/processed.csv'

# Load the CSV file into a pandas DataFrame
data = pd.read_csv(file_path)

# Print the number of rows before cleaning
print(f"Number of rows before cleaning: {len(data)}")

# Drop rows where "Previous Close" is missing
data = data.dropna(subset=["Previous Close"])

# Drop the article text columns (Article 1 to Article 14)
columns_to_drop = [f'Article {i}' for i in range(1, 15)]
data = data.drop(columns=columns_to_drop, errors='ignore')

# Fill missing values in sentiment_score_article_1 to sentiment_score_article_14 with 0
sentiment_columns = [f'sentiment_score_article_{i}' for i in range(1, 15)]
data[sentiment_columns] = data[sentiment_columns].fillna(0)

# Drop any rows that still have missing values
data = data.dropna()

# Convert boolean columns to integers (True -> 1, False -> 0)
bool_columns = data.select_dtypes(include=['bool']).columns
data[bool_columns] = data[bool_columns].astype(int)

# Drop 'ticker' and 'date_and_time' columns
data = data.drop(columns=['ticker', 'date_and_time'], errors='ignore')

# Function to split range columns into two numeric columns (min and max)
def split_range_column(df, column):
    """Splits a range column like 'low - high' into two separate numeric columns."""
    ranges = df[column].str.split(' - ', expand=True)
    df[f'{column}_min'] = pd.to_numeric(ranges[0], errors='coerce')
    df[f'{column}_max'] = pd.to_numeric(ranges[1], errors='coerce')
    df.drop(columns=[column], inplace=True)

# Process 'Day\'s Range' and '52 Week Range' columns
if "Day's Range" in data.columns:
    split_range_column(data, "Day's Range")
if "52 Week Range" in data.columns:
    split_range_column(data, "52 Week Range")

# Drop categorical columns
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

# Print the number of rows after final cleaning
print(f"Number of rows after final cleaning: {len(data)}")

# Save the prepared dataset to a new CSV file
prepared_file_path = 'data/prepared_for_model.csv'
data.to_csv(prepared_file_path, index=False)

print(f"Prepared dataset saved to {prepared_file_path}")