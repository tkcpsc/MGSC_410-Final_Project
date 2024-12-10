import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import time
import matplotlib.pyplot as plt

# Load and preprocess the dataset
file_path = 'data/prepared_for_model.csv'  # Adjust the file path as needed
data = pd.read_csv(file_path)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Define features and target
target_column = 't+1'
X = data.drop(columns=[target_column])
y = data[target_column]

# Ensure features are numeric
X = X.select_dtypes(include=['int64', 'float64', 'bool'])

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Rolling window function
def rolling_window_prediction(X, y, window_size=100, n_estimators=100, max_depth=5, learning_rate=0.1, reg_alpha=1.0, reg_lambda=1.0):
    """
    Perform rolling window predictions for time-series data.

    Args:
        X (np.ndarray): Scaled feature matrix.
        y (pd.Series): Target variable.
        window_size (int): Number of rows in the training window.
        n_estimators (int): Number of estimators for XGBoost.
        max_depth (int): Maximum depth of the trees.
        learning_rate (float): Learning rate for XGBoost.
        reg_alpha (float): L1 regularization term.
        reg_lambda (float): L2 regularization term.

    Returns:
        dict: Evaluation metrics and predictions.
    """
    predictions = []
    actuals = []
    start_time = time.time()
    saved_model = None  # Variable to store the last trained model

    print(f"Starting rolling window predictions with window size {window_size}...")

    for i in range(window_size, len(X)):
        # Define the rolling training and test sets
        X_train = X[i - window_size:i]
        y_train = y.iloc[i - window_size:i]
        X_test = X[[i]]
        y_test = y.iloc[i]

        # Train the model
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

        # Save the last trained model
        if i == len(X) - 1:
            saved_model = model

        # Make a prediction for the test row
        y_pred = model.predict(X_test)[0]
        predictions.append(y_pred)
        actuals.append(y_test)

        # Log progress
        if i % 100 == 0 or i == len(X) - 1:
            elapsed_time = time.time() - start_time
            print(f"Processed row {i}/{len(X)}. Time elapsed: {elapsed_time:.2f}s")

    # Save the last trained model
    if saved_model:
        saved_model.save_model("trained_xgb_model.json")
        print("Model saved as 'trained_xgb_model.json'.")

    # Calculate evaluation metrics
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print("Rolling window prediction completed.")
    print(f"Total time: {time.time() - start_time:.2f}s")

    return {
        "MSE": mse,
        "R2": r2,
        "Predictions": predictions,
        "Actuals": actuals,
    }

# Run the rolling window prediction
window_size = 100  # Adjust the window size as needed
results = rolling_window_prediction(
    X_scaled, y,
    window_size=window_size,
    n_estimators=150,  # Increase estimators for better model fitting
    max_depth=6,       # Allow slightly deeper trees
    learning_rate=0.05,  # Reduce learning rate for finer updates
    reg_alpha=2.0,       # Increase L1 regularization
    reg_lambda=3.0,      # Increase L2 regularization
)

# Print evaluation metrics
print("\nRolling Window Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {results['MSE']:.4f}")
print(f"R-Squared (R2): {results['R2']:.4f}")

# Visualize predictions vs actuals
plt.figure(figsize=(12, 6))
plt.plot(results['Actuals'], label='Actuals', linestyle='-', marker='o', alpha=0.6, markersize=4)
plt.plot(results['Predictions'], label='Predictions', linestyle='--', marker='x', alpha=0.6, markersize=4)
plt.legend()
plt.title("Optimized Rolling Window Predictions vs Actuals")
plt.xlabel("Time Step")
plt.ylabel("Target Value")
plt.tight_layout()
plt.show()
