# Problem Statement:
# Normalize the 'Math' scores using Min-Max scaling.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def main():
    df = pd.read_csv("students.csv")

    scaler = MinMaxScaler()
    df["Math_Normalized"] = scaler.fit_transform(df[["Math"]])

    print(df)


if __name__ == "__main__":
    main()