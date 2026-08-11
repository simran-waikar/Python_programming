# Problem Statement:
# Using pandas functions, calculate and display:
# 1. Average StudyHours
# 2. Average Attendance
# 3. Maximum PreviousScore
# 4. Minimum SleepHours

import pandas as pd


def main():
    # Load the dataset
    df = pd.read_csv("student_performance_ml.csv")

    # Calculate required values
    average_study_hours = df["StudyHours"].mean()
    average_attendance = df["Attendance"].mean()
    maximum_previous_score = df["PreviousScore"].max()
    minimum_sleep_hours = df["SleepHours"].min()

    # Display results
    print("Average Study Hours:", average_study_hours)
    print("Average Attendance:", average_attendance)
    print("Maximum Previous Score:", maximum_previous_score)
    print("Minimum Sleep Hours:", minimum_sleep_hours)


if __name__ == "__main__":
    main()