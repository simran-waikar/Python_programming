# Problem Statement:
# Write a Python program to:
# 1. Display the total number of students in the dataset.
# 2. Count how many students Passed (FinalResult = 1).
# 3. Count how many students Failed (FinalResult = 0).

import pandas as pd


def main():
    # Load the dataset
    df = pd.read_csv("student_performance_ml.csv")

    # Total number of students
    total_students = len(df)

    # Count passed students
    passed_students = (df["FinalResult"] == 1).sum()

    # Count failed students
    failed_students = (df["FinalResult"] == 0).sum()

    print("Total Number of Students:", total_students)
    print("Number of Passed Students:", passed_students)
    print("Number of Failed Students:", failed_students)


if __name__ == "__main__":
    main()