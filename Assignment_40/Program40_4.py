# Problem Statement:
# Create a new DataFrame containing details of 5 new students.
# Use the trained Decision Tree model to predict their FinalResult.
# Display the predictions clearly.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():

    # Load dataset
    data = pd.read_csv("student_performance_ml.csv")

    # Separate features and target
    X = data.drop("FinalResult", axis=1)
    y = data["FinalResult"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Create data for 5 new students
    new_students = pd.DataFrame({
        "StudyHours": [2, 5, 7, 3, 8],
        "Attendance": [65, 80, 92, 70, 95],
        "PreviousScore": [45, 65, 85, 50, 90],
        "AssignmentsCompleted": [4, 7, 9, 5, 10],
        "SleepHours": [6, 7, 8, 6, 8]
    })

    # Predict
    predictions = model.predict(new_students)

    # Add prediction column
    new_students["PredictedResult"] = predictions

    # Convert 1/0 into Pass/Fail
    new_students["PredictedResult"] = new_students[
        "PredictedResult"
    ].map({1: "Pass", 0: "Fail"})

    print("Predictions for New Students:")
    print("--------------------------------")
    print(new_students.to_string(index=False))


if __name__ == "__main__":
    main()