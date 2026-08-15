# Problem Statement:
# Add a new column 'Status' where students with total >= 250
# are 'Pass', else 'Fail'.

import pandas as pd


def main():
    df = pd.read_csv("students.csv")

    # Calculate total marks
    df["Total"] = df[["Math", "English", "Science"]].sum(axis=1)

    # Add Status column
    df["Status"] = df["Total"].apply(
        lambda total: "Pass" if total >= 250 else "Fail"
    )

    print(df)


if __name__ == "__main__":
    main()