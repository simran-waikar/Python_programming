# Problem Statement:
# Identify students where:
# y_test != y_pred
#
# Display those rows, count how many students were misclassified,
# and observe the common pattern among the misclassified students.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


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

    # Predict
    y_pred = model.predict(X_test)

    # Find misclassified students
    misclassified = X_test.copy()

    misclassified["ActualResult"] = y_test
    misclassified["PredictedResult"] = y_pred

    misclassified = misclassified[
        misclassified["ActualResult"] != misclassified["PredictedResult"]
    ]

    # Display misclassified students
    print("Misclassified Students:")
    print("-----------------------")

    if len(misclassified) == 0:
        print("No students were misclassified.")
    else:
        print(misclassified.to_string())

    # Number of misclassified students
    print("\nNumber of Misclassified Students:",
          len(misclassified))

    # Common pattern
    if len(misclassified) > 0:
        print("\nPossible Common Pattern:")
        print("Misclassified students may have similar or borderline")
        print("values in features such as StudyHours, Attendance,")
        print("PreviousScore and AssignmentsCompleted.")


if __name__ == "__main__":
    main()