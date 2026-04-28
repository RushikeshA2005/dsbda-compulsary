"""
Linear Regression Model to predict Sales based on Discount, Quantity, and Profit.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # to save the model

def train_sales_model(data_path="clean_superstore.csv", model_path="sales_model.pkl"):
    df = pd.read_csv(data_path)
    
    # Select features and target
    X = df[["Discount", "Quantity", "Profit"]]
    y = df["Sales"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Performance:")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.4f}")
    
    print("\nCoefficients:")
    for feat, coef in zip(X.columns, model.coef_):
        print(f"  {feat}: {coef:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    
    # Save model
    joblib.dump(model, model_path)
    print(f"\nModel saved as {model_path}")
    
    return model

if __name__ == "__main__":
    train_sales_model()
