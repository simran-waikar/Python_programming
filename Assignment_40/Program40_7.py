# Problem Statement:
# Train the Decision Tree model using:
# random_state = 0
# random_state = 10
# random_state = 42
#
# Compare the testing accuracy and determine whether the result changes.

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

    random_states = [0, 10, 42]

    print("Testing Accuracy for Different random_state Values")
    print("---------------------------------------------------")

    for state in random_states:

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=state
        )

        # Train model
        model = DecisionTreeClassifier(random_state=state)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print(f"random_state = {state}  -->  Accuracy = {accuracy:.4f}")

    print("\nConclusion:")
    print("Changing random_state can change the train-test split")
    print("and therefore may change the testing accuracy.")


if __name__ == "__main__":
    main()