# Problem Statement 7:
# Use the trained model to predict the result for a student with:
#
# StudyHours = 6
# Attendance = 85
# PreviousScore = 66
# AssignmentsCompleted = 7
# SleepHours = 7
#
# Determine whether the student will Pass or Fail.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    # Load dataset
    data = pd.read_csv("student_performance_ml.csv")

    # Define features and target
    X = data[[
        "StudyHours",
        "Attendance",
        "PreviousScore",
        "AssignmentsCompleted",
        "SleepHours"
    ]]

    y = data["FinalResult"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Create data for the new student
    student = pd.DataFrame([{
        "StudyHours": 6,
        "Attendance": 85,
        "PreviousScore": 66,
        "AssignmentsCompleted": 7,
        "SleepHours": 7
    }])

    # Predict result
    prediction = model.predict(student)

    # Display result
    if prediction[0] == 1:
        print("Prediction: PASS")
    else:
        print("Prediction: FAIL")


if __name__ == "__main__":
    main()