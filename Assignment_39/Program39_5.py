# Problem Statement 5:
# Calculate:
# 1. Training accuracy
# 2. Testing accuracy
# Compare both and comment whether the model is overfitting or underfitting.

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

    # Create and train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calculate accuracies
    training_accuracy = accuracy_score(y_train, train_pred)
    testing_accuracy = accuracy_score(y_test, test_pred)

    # Display results
    print("Training Accuracy:", training_accuracy * 100, "%")
    print("Testing Accuracy :", testing_accuracy * 100, "%")

    # Compare accuracies
    difference = training_accuracy - testing_accuracy

    print("\nModel Analysis:")

    if difference > 0.15:
        print("The model is likely OVERFITTING.")
        print("Training accuracy is much higher than testing accuracy.")

    elif training_accuracy < 0.70 and testing_accuracy < 0.70:
        print("The model may be UNDERFITTING.")
        print("Both training and testing accuracies are low.")

    else:
        print("The model appears to have a reasonable fit.")
        print("Training and testing accuracies are relatively close.")


if __name__ == "__main__":
    main()