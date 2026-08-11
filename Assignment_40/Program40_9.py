# Problem Statement:
# Create a new column:
#
# PerformanceIndex = (StudyHours * 2) + Attendance
#
# Train the Decision Tree model including this new feature.
# Determine whether the accuracy improves.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def main():

    # Load dataset
    data = pd.read_csv("student_performance_ml.csv")

    # ---------------------------------------
    # Original model
    # ---------------------------------------

    X_original = data.drop("FinalResult", axis=1)
    y = data["FinalResult"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_original, y, test_size=0.2, random_state=42
    )

    original_model = DecisionTreeClassifier(random_state=42)
    original_model.fit(X_train, y_train)

    original_pred = original_model.predict(X_test)

    original_accuracy = accuracy_score(y_test, original_pred)

    # ---------------------------------------
    # Create PerformanceIndex
    # ---------------------------------------

    data["PerformanceIndex"] = (
        data["StudyHours"] * 2
    ) + data["Attendance"]

    # Features including new column
    X_new = data.drop("FinalResult", axis=1)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X_new, y, test_size=0.2, random_state=42
    )

    new_model = DecisionTreeClassifier(random_state=42)
    new_model.fit(X_train2, y_train2)

    new_pred = new_model.predict(X_test2)

    new_accuracy = accuracy_score(y_test2, new_pred)

    # Display results
    print("Original Accuracy:", original_accuracy)
    print("Accuracy with PerformanceIndex:", new_accuracy)

    if new_accuracy > original_accuracy:
        print("\nAccuracy improved after adding PerformanceIndex.")
    elif new_accuracy < original_accuracy:
        print("\nAccuracy decreased after adding PerformanceIndex.")
    else:
        print("\nAccuracy remained the same.")


if __name__ == "__main__":
    main()