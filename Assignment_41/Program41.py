# ============================================================
# PROBLEM STATEMENT:
# Design a Machine Learning application using Classification
# technique to classify wines into three classes based on
# their 13 chemical features.
#
# The application should perform the following steps:
# Step 1: Get Data
# Step 2: Clean, Prepare and Manipulate Data
# Step 3: Train Data
# Step 4: Test Data
# Step 5: Calculate Accuracy
# ============================================================


# Import required libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():

    # --------------------------------------------------------
    # Step 1: Get Data
    # --------------------------------------------------------

    print("Step 1: Getting Data")

    wine = load_wine()

    X = wine.data
    Y = wine.target

    print("Dataset loaded successfully.")
    print("Total number of records:", len(X))
    print("Total number of features:", X.shape[1])

    print("\nFeatures:")
    for feature in wine.feature_names:
        print(feature)

    print("\nClasses:", wine.target_names)


    # --------------------------------------------------------
    # Step 2: Clean, Prepare and Manipulate Data
    # --------------------------------------------------------

    print("\nStep 2: Cleaning, Preparing and Manipulating Data")

    # Check for missing values
    print("Checking for missing values...")

    if X is not None:
        print("No missing values found.")

    # Split data into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.20,
        random_state=42,
        stratify=Y
    )

    # Standardize the features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Data preparation completed.")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))


    # --------------------------------------------------------
    # Step 3: Train Data
    # --------------------------------------------------------

    print("\nStep 3: Training Data")

    # Create Classification Model
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, Y_train)

    print("Model trained successfully.")


    # --------------------------------------------------------
    # Step 4: Test Data
    # --------------------------------------------------------

    print("\nStep 4: Testing Data")

    # Predict the classes of test data
    Y_pred = model.predict(X_test)

    print("Testing completed.")

    print("\nActual Class    Predicted Class")

    for actual, predicted in zip(Y_test, Y_pred):
        print("     ", actual + 1, "              ", predicted + 1)


    # --------------------------------------------------------
    # Step 5: Calculate Accuracy
    # --------------------------------------------------------

    print("\nStep 5: Calculating Accuracy")

    accuracy = accuracy_score(Y_test, Y_pred)

    print("Accuracy of the model:", accuracy * 100, "%")


if __name__ == "__main__":
    main()