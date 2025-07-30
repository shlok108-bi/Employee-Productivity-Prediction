# Employee_Prediction.py
# Modular implementation of data analysis, preprocessing, and model building for employee performance prediction

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from utils import MultiColumnLabelEncoder

# --- Data Collection ---
# Load the dataset
data = pd.read_csv('garments_worker_productivity.csv')

# --- Descriptive Analysis ---
# Display basic information about the dataset
print("Dataset Info:")
print(data.info())
print("\nDataset Shape:", data.shape)
print("\nDescriptive Statistics:")
print(data.describe(include='all'))

# --- Correlation Analysis ---
# Select numerical columns for correlation matrix
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = data[numerical_cols].corr()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# --- Data Preprocessing ---
# Check for null values
print("\nNull Values in Dataset:")
print(data.isnull().sum())

# Drop 'wip' column due to high number of missing values
data = data.drop(columns=['wip'])

# Handle date column: Convert to datetime and extract month
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month
data = data.drop(columns=['date'])

# Handle department column: Merge 'finishing ' and 'finishing'
data['department'] = data['department'].str.strip().replace('finishing ', 'finishing')

# Handle categorical data: Encode 'quarter', 'department', 'day'
categorical_cols = ['quarter', 'department', 'day']
mcle = MultiColumnLabelEncoder(columns=categorical_cols)
data = mcle.fit_transform(data)

# Save the encoder for later use in Flask app
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(mcle, f)

# --- Split Data into Train and Test ---
# Define features (X) and target (y)
X = data.drop(columns=['actual_productivity'])
y = data['actual_productivity']

# Convert X to array
X = X.values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Building ---
# Initialize models
model_lr = LinearRegression()
model_rf = RandomForestRegressor(random_state=42)
model_xgb = XGBRegressor(random_state=42)

# Train models
model_lr.fit(X_train, y_train)
model_rf.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)

# Make predictions
pred_lr = model_lr.predict(X_test)
pred_rf = model_rf.predict(X_test)
pred_xgb = model_xgb.predict(X_test)

# --- Model Evaluation ---
# Function to evaluate and print metrics
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    return mae, mse, r2

# Evaluate all models
lr_metrics = evaluate_model(y_test, pred_lr, "Linear Regression")
rf_metrics = evaluate_model(y_test, pred_rf, "Random Forest")
xgb_metrics = evaluate_model(y_test, pred_xgb, "XGBoost")

# Compare models and select the best based on R2 score
metrics = {
    'Linear Regression': lr_metrics,
    'Random Forest': rf_metrics,
    'XGBoost': xgb_metrics
}
best_model_name = max(metrics, key=lambda x: metrics[x][2])
print(f"\nBest Model: {best_model_name}")

# --- Save the Best Model ---
models = {'Linear Regression': model_lr, 'Random Forest': model_rf, 'XGBoost': model_xgb}
with open('gwp.pkl', 'wb') as f:
    pickle.dump(models[best_model_name], f)
print(f"\nBest model ({best_model_name}) saved as 'gwp.pkl'")