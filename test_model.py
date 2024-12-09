import pandas as pd
from xgboost import XGBRegressor

# Load the pretrained model
model_file_path = 'trained_xgb_model.json'
model = XGBRegressor()
model.load_model(model_file_path)

# Load the testing data
testing_data_file = 'data/user_input_cleaned.csv'
user_data = pd.read_csv(testing_data_file)

# Drop the 't+1' column if it exists
if 't+1' in user_data.columns:
    user_data = user_data.drop(columns=['t+1'], errors='ignore')

# Select numeric and boolean columns
X_user = user_data.select_dtypes(include=['int64', 'float64', 'bool'])

# Handle missing values by filling with column mean
X_user = X_user.fillna(X_user.mean())

# Predict the t+1 price
predicted_t_plus_1 = model.predict(X_user)

# Print the predicted output
if len(predicted_t_plus_1) > 0:
    print(f"Predicted t+1 price: {predicted_t_plus_1[0]}")
else:
    raise ValueError("Prediction array is empty.")