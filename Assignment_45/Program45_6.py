# Problem Statement:
# Count how many students passed.

import pandas as pd


def main():
    df = pd.read_csv("students.csv")

    # Calculate total marks
    df["Total"] = df[["Math", "English", "Science"]].sum(axis=1)

    # Add Status column
    df["Status"] = df["Total"].apply(
        lambda total: "Pass" if total >= 250 else "Fail"
    )

    passed_students = (df["Status"] == "Pass").sum()

    print("Number of students passed:", passed_students)


if __name__ == "__main__":
    main()