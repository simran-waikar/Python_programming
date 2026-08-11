# Problem Statement:
# Without using accuracy_score, manually calculate accuracy.
# Verify whether the manually calculated accuracy matches
# the accuracy calculated by sklearn.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


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

    # Predictions
    y_pred = model.predict(X_test)

    # ---------------------------------
    # Manual accuracy calculation
    # ---------------------------------

    correct = 0

    for actual, predicted in zip(y_test, y_pred):
        if actual == predicted:
            correct += 1

    total = len(y_test)

    manual_accuracy = correct / total

    # sklearn accuracy
    sklearn_accuracy = accuracy_score(y_test, y_pred)

    print("Total Test Students:", total)
    print("Correct Predictions:", correct)

    print("\nManual Accuracy:", manual_accuracy)
    print("Sklearn Accuracy:", sklearn_accuracy)

    if manual_accuracy == sklearn_accuracy:
        print("\nBoth accuracy values match.")
    else:
        print("\nAccuracy values do not match.")


if __name__ == "__main__":
    main()