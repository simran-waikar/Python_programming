# Problem Statement 2:
# Use the trained model to predict results for X_test.
# Display predicted values along with actual values.

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

    # Predict values
    y_pred = model.predict(X_test)

    # Display actual and predicted values
    print("Actual Values    Predicted Values")
    print("--------------------------------")

    for actual, predicted in zip(y_test, y_pred):
        print(f"{actual:<17} {predicted}")


if __name__ == "__main__":
    main()