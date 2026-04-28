"""
Data Cleaning for Superstore Sales Dataset
Handles missing values, duplicates, outliers, and data type conversion.
"""

import pandas as pd
import numpy as np

def clean_superstore_data(input_path="superstore.csv", output_path="clean_superstore.csv"):
    # Load dataset (adjust encoding if needed)
    df = pd.read_csv(input_path, encoding="latin1")
    
    print("Initial shape:", df.shape)
    print("Missing values per column:\n", df.isnull().sum())
    
    # Drop rows with missing Postal Code (only missing column)
    df.dropna(subset=["Postal Code"], inplace=True)
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    # Remove negative sales (data errors)
    df = df[df["Sales"] >= 0]
    
    # Remove outliers in Sales using IQR method
    Q1 = df["Sales"].quantile(0.25)
    Q3 = df["Sales"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df["Sales"] >= lower) & (df["Sales"] <= upper)]
    
    # Convert date columns to datetime
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print("Final shape:", df.shape)
    return df

if __name__ == "__main__":
    clean_superstore_data()
