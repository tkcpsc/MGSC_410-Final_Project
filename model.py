import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# Load the prepared dataset
file_path = 'data/prepared_for_model.csv'
data = pd.read_csv(file_path)

# Define the target column and features
target_column = 't+1'

# Split features and target
X = data.drop(columns=[target_column], errors='ignore')
y = data[target_column]

# Ensure all columns are numeric or boolean
X = X.select_dtypes(include=['int64', 'float64', 'bool'])

# Replace infinity values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values
X.dropna(inplace=True)
y = y[X.index]  # Align target variable with cleaned features

# Ensure values are finite
if not np.isfinite(X).all().all():
    raise ValueError("Features contain non-finite values (NaN, infinity, or overly large values).")

if not np.isfinite(y).all():
    raise ValueError("Target contains non-finite values (NaN, infinity, or overly large values).")

# Create a time-based train-test split
train_size = int(len(X) * 0.8)  # Use 80% for training
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Initialize the XGBoost model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Save the trained model
model_file_path = 'trained_xgb_model.json'
model.save_model(model_file_path)
print(f"Model saved to {model_file_path}")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-Squared (R2 Score): {r2}")