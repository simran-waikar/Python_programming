# Problem Statement 6:
# Train three Decision Tree models with:
# 1. max_depth = 1
# 2. max_depth = 3
# 3. max_depth = None
#
# Compare their testing accuracies and write observations.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


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

    # Create three Decision Tree models
    model_1 = DecisionTreeClassifier(max_depth=1, random_state=42)
    model_3 = DecisionTreeClassifier(max_depth=3, random_state=42)
    model_none = DecisionTreeClassifier(max_depth=None, random_state=42)

    # Train models
    model_1.fit(X_train, y_train)
    model_3.fit(X_train, y_train)
    model_none.fit(X_train, y_train)

    # Make predictions
    pred_1 = model_1.predict(X_test)
    pred_3 = model_3.predict(X_test)
    pred_none = model_none.predict(X_test)

    # Calculate testing accuracies
    accuracy_1 = accuracy_score(y_test, pred_1)
    accuracy_3 = accuracy_score(y_test, pred_3)
    accuracy_none = accuracy_score(y_test, pred_none)

    # Display results
    print("Testing Accuracies:")
    print("-------------------")
    print("max_depth = 1   :", accuracy_1 * 100, "%")
    print("max_depth = 3   :", accuracy_3 * 100, "%")
    print("max_depth = None:", accuracy_none * 100, "%")

    # Find best model
    accuracies = {
        "max_depth = 1": accuracy_1,
        "max_depth = 3": accuracy_3,
        "max_depth = None": accuracy_none
    }

    best_model = max(accuracies, key=accuracies.get)

    print("\nObservation:")
    print("The model with the highest testing accuracy is:", best_model)
    print("A very deep tree may overfit the training data.")
    print("A very shallow tree may underfit the data.")


if __name__ == "__main__":
    main()