# Problem Statement:
# Based on the dataset values, analyze whether:
# 1. Higher StudyHours increase the chance of passing.
# 2. Higher Attendance improves FinalResult.
# Write observations in 4-5 lines.

import pandas as pd


def main():
    # Load the dataset
    df = pd.read_csv("student_performance_ml.csv")

    # Calculate average StudyHours for Pass and Fail
    study_hours = df.groupby("FinalResult")["StudyHours"].mean()

    # Calculate average Attendance for Pass and Fail
    attendance = df.groupby("FinalResult")["Attendance"].mean()

    print("Average Study Hours:")
    print(study_hours)

    print("\nAverage Attendance:")
    print(attendance)

    print("\nObservations:")

    if study_hours[1] > study_hours[0]:
        print("1. Students who passed generally studied for more hours.")
    else:
        print("1. Higher study hours do not show a clear increase in passing.")

    if attendance[1] > attendance[0]:
        print("2. Passed students generally have higher attendance.")
    else:
        print("2. Higher attendance does not show a clear improvement in results.")

    print("3. StudyHours and Attendance can be useful factors for predicting student performance.")
    print("4. However, passing depends on multiple factors and not only study hours or attendance.")
    print("5. The exact relationship can be further studied using machine learning.")


if __name__ == "__main__":
    main()