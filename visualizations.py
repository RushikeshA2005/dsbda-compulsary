"""
Generate all required charts for the dashboard.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_all_charts(data_path="clean_superstore.csv"):
    df = pd.read_csv(data_path)
    
    # Ensure Order Date is datetime
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    
    # 1. Bar chart: Sales by Category
    plt.figure(figsize=(8,5))
    cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    cat_sales.plot(kind="bar", color="steelblue")
    plt.title("Total Sales by Product Category")
    plt.ylabel("Sales ($)")
    plt.xlabel("Category")
    plt.tight_layout()
    plt.savefig("bar_chart_category.png")
    plt.close()
    
    # 2. Line graph: Monthly revenue trend
    plt.figure(figsize=(10,5))
    monthly = df.set_index("Order Date").resample("M")["Sales"].sum()
    monthly.plot(marker="o", color="green")
    plt.title("Monthly Revenue Trend")
    plt.ylabel("Revenue ($)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("line_monthly_revenue.png")
    plt.close()
    
    # 3. Pie chart: Sales by Region
    plt.figure(figsize=(7,7))
    region_sales = df.groupby("Region")["Sales"].sum()
    region_sales.plot(kind="pie", autopct="%1.1f%%", startangle=90)
    plt.title("Sales Distribution by Region")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("pie_region.png")
    plt.close()
    
    # 4. Heatmap: Correlation between numeric features
    plt.figure(figsize=(8,6))
    numeric_cols = ["Sales", "Quantity", "Discount", "Profit"]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("heatmap_correlation.png")
    plt.close()
    
    # 5. Scatter plot: Discount vs Sales
    plt.figure(figsize=(8,5))
    plt.scatter(df["Discount"], df["Sales"], alpha=0.5, color="purple")
    plt.xlabel("Discount")
    plt.ylabel("Sales ($)")
    plt.title("Discount vs Sales Relationship")
    plt.tight_layout()
    plt.savefig("scatter_discount_sales.png")
    plt.close()
    
    print("All charts saved as PNG files.")

if __name__ == "__main__":
    generate_all_charts()
