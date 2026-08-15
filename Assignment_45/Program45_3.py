# Problem Statement:
# Group students by gender and calculate average marks.

import pandas as pd


def main():
    df = pd.read_csv("students.csv")

    # Add gender values according to the students in your dataset
    gender = ["Male", "Female", "Male", "Female", "Male"]

    df["Gender"] = gender[:len(df)]

    average_marks = df.groupby("Gender")[["Math", "English", "Science"]].mean()

    print("Average marks by gender:")
    print(average_marks)


if __name__ == "__main__":
    main()