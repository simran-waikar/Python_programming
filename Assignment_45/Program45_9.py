# Problem Statement:
# Rename 'Math' column to 'Mathematics'.

import pandas as pd


def main():
    df = pd.read_csv("students.csv")

    df.rename(columns={"Math": "Mathematics"}, inplace=True)

    print(df)


if __name__ == "__main__":
    main()