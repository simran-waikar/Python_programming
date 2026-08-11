# Problem Statement:
# Write a Python program to load the file student_performance_ml.csv using pandas.
# Display:
# 1. First 5 records
# 2. Last 5 records
# 3. Total number of rows and columns
# 4. List of column names
# 5. Data types of each column

import pandas as pd


def main():
    # Load the dataset
    df = pd.read_csv("student_performance_ml.csv")

    # Display first 5 records
    print("First 5 Records:")
    print(df.head())

    # Display last 5 records
    print("\nLast 5 Records:")
    print(df.tail())

    # Display number of rows and columns
    print("\nTotal Rows and Columns:")
    print(df.shape)

    # Display column names
    print("\nColumn Names:")
    print(df.columns.tolist())

    # Display data types
    print("\nData Types of Each Column:")
    print(df.dtypes)


if __name__ == "__main__":
    main()