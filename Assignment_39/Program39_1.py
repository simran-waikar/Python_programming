# Problem Statement 1:
# Import DecisionTreeClassifier from sklearn.
# Create a model object and train it using fit().

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    # Load the dataset
    data = pd.read_csv("student_performance_ml.csv")

    # Separate input features and target variable
    X = data[[
        "StudyHours",
        "Attendance",
        "PreviousScore",
        "AssignmentsCompleted",
        "SleepHours"
    ]]

    y = data["FinalResult"]

    # Split the dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create Decision Tree model
    model = DecisionTreeClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    print("Decision Tree model trained successfully.")


if __name__ == "__main__":
    main()