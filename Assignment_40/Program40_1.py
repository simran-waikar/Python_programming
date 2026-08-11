# Problem Statement:
# After training the Decision Tree model, use model.feature_importances_
# to display the importance score of each feature.
# Find which feature contributes the most and the least in predicting FinalResult.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():

    # Load dataset
    data = pd.read_csv("student_performance_ml.csv")

    # Separate input and output
    X = data.drop("FinalResult", axis=1)
    y = data["FinalResult"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train Decision Tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Get feature importance
    importance = model.feature_importances_

    print("Feature Importance:")
    print("-------------------")

    for feature, score in zip(X.columns, importance):
        print(f"{feature}: {score:.4f}")

    # Find most and least important features
    most_important = X.columns[importance.argmax()]
    least_important = X.columns[importance.argmin()]

    print("\nMost Important Feature:", most_important)
    print("Least Important Feature:", least_important)


if __name__ == "__main__":
    main()