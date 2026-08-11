# Problem Statement:
# Train the Decision Tree model with:
#
# max_depth = None
#
# Calculate:
# 1. Training accuracy
# 2. Testing accuracy
#
# If training accuracy is 100% but testing accuracy is lower,
# explain why this happens.

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

    # Train Decision Tree with unlimited depth
    model = DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calculate accuracy
    training_accuracy = accuracy_score(y_train, train_pred)
    testing_accuracy = accuracy_score(y_test, test_pred)

    # Display results
    print("Training Accuracy:", training_accuracy)
    print("Testing Accuracy:", testing_accuracy)

    # Explanation
    print("\nExplanation:")

    if training_accuracy == 1.0 and testing_accuracy < 1.0:
        print("The model may be overfitting the training data.")
        print("Since max_depth=None allows the Decision Tree to grow")
        print("without a depth limit, it can learn the training data")
        print("very closely, including noise and small patterns.")
        print("Therefore, it can achieve 100% training accuracy while")
        print("performing worse on unseen testing data.")
    else:
        print("The model does not show a clear case of 100%")
        print("training accuracy with lower testing accuracy.")


if __name__ == "__main__":
    main()