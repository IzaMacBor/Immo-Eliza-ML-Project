import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from joblib import dump
import os

# Define paths
data_path = r'...\data\standardized_data.csv'
results_path = r'...\results'

# Create results directory if it does not exist
os.makedirs(results_path, exist_ok=True)

# Load preprocessed data
df = pd.read_csv(data_path)

# Define feature matrix (X) and target variable (y)
X = df.drop(columns=['price'])
y = df['price']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Random Forest Regressor
model = RandomForestRegressor(random_state=42)

# Set up GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=model, 
    param_grid=param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error', 
    n_jobs=-1
)

# Fit the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_model = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)

# Evaluate model performance on the test set
y_pred_test = best_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\nTest Set Performance:")
print(f'MSE: {mse_test:.4f}, MAE: {mae_test:.4f}, R2 Score: {r2_test:.4f}')

# Evaluate model performance on the training set
y_pred_train = best_model.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

print("\nTrain Set Performance:")
print(f'MSE: {mse_train:.4f}, MAE: {mae_train:.4f}, R2 Score: {r2_train:.4f}')

# Perform cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()
print(f"\nCross-Validation MSE: {cv_mse:.4f}")

# Save the model with a constant name to always have the latest version
model_file = os.path.join(results_path, 'random_forest_model.joblib')
dump(best_model, model_file)
print(f"\nModel saved to: {model_file}")

# Save results to CSV
results_file = os.path.join(results_path, 'random_forest_results.csv')
results_df = pd.DataFrame([{
    'Model': 'Random Forest',
    'Best Parameters': grid_search.best_params_,
    'Test MSE': mse_test,
    'Test MAE': mae_test,
    'Test R2': r2_test,
    'Train MSE': mse_train,
    'Train MAE': mae_train,
    'Train R2': r2_train,
    'Cross-Validation MSE': cv_mse
}])

results_df.to_csv(results_file, index=False)
print(f"Results saved to: {results_file}")
