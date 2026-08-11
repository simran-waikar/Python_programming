# Problem Statement:
# Train the Decision Tree model using only:
# 1. StudyHours
# 2. Attendance
# Compare the accuracy with the full-feature model.
# Determine whether the model is still performing well.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def main():

    # Load dataset
    data = pd.read_csv("student_performance_ml.csv")

    y = data["FinalResult"]

    # -----------------------------
    # Full-feature model
    # -----------------------------
    X_full = data.drop("FinalResult", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42
    )

    full_model = DecisionTreeClassifier(random_state=42)
    full_model.fit(X_train, y_train)

    full_pred = full_model.predict(X_test)

    full_accuracy = accuracy_score(y_test, full_pred)

    # -----------------------------
    # Model using only two features
    # -----------------------------
    X_two = data[["StudyHours", "Attendance"]]

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X_two, y, test_size=0.2, random_state=42
    )

    two_feature_model = DecisionTreeClassifier(random_state=42)
    two_feature_model.fit(X_train2, y_train2)

    two_pred = two_feature_model.predict(X_test2)

    two_feature_accuracy = accuracy_score(y_test2, two_pred)

    # Display results
    print("Full-Feature Model Accuracy:", full_accuracy)
    print("StudyHours + Attendance Accuracy:", two_feature_accuracy)

    if two_feature_accuracy >= full_accuracy:
        print("\nThe model performs well using only StudyHours and Attendance.")
    else:
        print("\nThe accuracy decreases when only two features are used.")


if __name__ == "__main__":
    main()