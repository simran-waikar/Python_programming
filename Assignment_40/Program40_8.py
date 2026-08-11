# Problem Statement:
# Visualize the trained Decision Tree using plot_tree.
# Identify which feature appears at the root node and explain
# why that feature was selected first.

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


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

    # Train Decision Tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Find root feature
    root_feature_index = model.tree_.feature[0]
    root_feature = X.columns[root_feature_index]

    print("Root Node Feature:", root_feature)

    print("\nReason:")
    print("The root feature is selected because it provides the")
    print("best split of the training data according to the")
    print("Decision Tree's splitting criterion.")

    # Visualize tree
    plt.figure(figsize=(18, 10))

    plot_tree(
        model,
        feature_names=X.columns,
        class_names=["Fail", "Pass"],
        filled=True,
        rounded=True
    )

    plt.title("Decision Tree - Student Performance")
    plt.show()


if __name__ == "__main__":
    main()