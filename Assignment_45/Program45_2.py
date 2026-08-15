# Problem Statement:
# Create a gender column and perform one-hot encoding.

import pandas as pd


def main():
    df = pd.read_csv("students.csv")

    # Add gender values according to the students in your dataset
    gender = ["Male", "Female", "Male", "Female", "Male"]

    df["Gender"] = gender[:len(df)]

    # Perform one-hot encoding
    df = pd.get_dummies(df, columns=["Gender"])

    print(df)


if __name__ == "__main__":
    main()