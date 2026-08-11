# Problem Statement 4:
# Generate confusion matrix using sklearn.
# Display it using ConfusionMatrixDisplay.
# Explain True Positive, True Negative, False Positive and False Negative.

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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

    # Predict test values
    y_pred = model.predict(X_test)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    # Display confusion matrix
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fail", "Pass"]
    )

    display.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # Explanation
    print("\nExplanation:")
    print("True Positive (TP): Actual Pass and predicted Pass.")
    print("True Negative (TN): Actual Fail and predicted Fail.")
    print("False Positive (FP): Actual Fail but predicted Pass.")
    print("False Negative (FN): Actual Pass but predicted Fail.")


if __name__ == "__main__":
    main()