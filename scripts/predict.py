import pandas as pd
from joblib import load
import os
import json

# Load the trained model
model_path = r'...\results\random_forest_model.joblib'
model = load(model_path)

# Load new data for prediction
data_path = r'...\data\new_data.csv'
data = pd.read_csv(data_path)

# Ensure the data does not contain the target variable
X_new = data.drop(columns=['price'], errors='ignore')

# Generate predictions
predictions = model.predict(X_new)

# Load the mean and std from JSON file
with open(r'...\notebooks_preprocesing\price_stats.json', "r") as f:
    stats = json.load(f)

price_mean = stats["price_mean"]
price_std = stats["price_std"]


# Reverse the standardization of predictions
#price_mean = 345582.74354898534  # Substitute with the actual mean of the 'price' column before standardization
#price_std = 171005.6913476945    # Substitute with the actual std of the 'price' column before standardization
predictions_original = (predictions * price_std) + price_mean

# Save the original predictions to a new CSV file
output_path = r'...results\predictions.csv'
predictions_df = pd.DataFrame(predictions_original, columns=['Predicted Price'])
predictions_df.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}.")


