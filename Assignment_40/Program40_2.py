# Problem Statement:
# Remove the column SleepHours from the dataset.
# Train the Decision Tree model again and compare the new accuracy
# with the previous accuracy.
# Determine whether removing SleepHours affects performance.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def main():

    # Load dataset
    data = pd.read_csv("student_performance_ml.csv")

    # -----------------------------
    # Full-feature model
    # -----------------------------
    X = data.drop("FinalResult", axis=1)
    y = data["FinalResult"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    original_accuracy = accuracy_score(y_test, y_pred)

    # -----------------------------
    # Model without SleepHours
    # -----------------------------
    X_without_sleep = data.drop(
        ["FinalResult", "SleepHours"], axis=1
    )

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X_without_sleep, y, test_size=0.2, random_state=42
    )

    model2 = DecisionTreeClassifier(random_state=42)
    model2.fit(X_train2, y_train2)

    y_pred2 = model2.predict(X_test2)

    new_accuracy = accuracy_score(y_test2, y_pred2)

    # Display results
    print("Original Accuracy:", original_accuracy)
    print("Accuracy after removing SleepHours:", new_accuracy)

    if new_accuracy > original_accuracy:
        print("Performance improved after removing SleepHours.")
    elif new_accuracy < original_accuracy:
        print("Performance decreased after removing SleepHours.")
    else:
        print("Performance remained the same.")


if __name__ == "__main__":
    main()