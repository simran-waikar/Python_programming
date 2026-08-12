# ============================================================
# Problem Statement:
#
# Design a Machine Learning application using Classification
# technique to predict whether a person will Play or Not based
# on Weather and Temperature conditions.
#
# Features:
#   1. Weather
#   2. Temperature
#
# Labels:
#   Yes
#   No
#
# Algorithm:
#   K-Nearest Neighbour (KNN)
#
# K Value:
#   3
#
# ============================================================


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ------------------------------------------------------------
# Function: CheckAccuracy
# Purpose : Calculate accuracy of the KNN algorithm
# ------------------------------------------------------------
def CheckAccuracy(X, Y, k):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.5,
        random_state=42
    )

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)

    return accuracy


# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------
def main():

    # --------------------------------------------------------
    # Step 1: Get Data
    # --------------------------------------------------------

    print("Step 1: Get Data")
    print("----------------")

    data = pd.read_csv("MarvellousInfosystems_PlayPredictor.csv")

    print("Dataset:")
    print(data)
    print()


    # --------------------------------------------------------
    # Step 2: Clean, Prepare and Manipulate Data
    # --------------------------------------------------------

    print("Step 2: Clean, Prepare and Manipulate Data")
    print("-------------------------------------------")

    # Create LabelEncoder objects
    weather_encoder = LabelEncoder()
    temperature_encoder = LabelEncoder()
    play_encoder = LabelEncoder()

    # Convert Weather into numerical values
    data["Weather"] = weather_encoder.fit_transform(
        data["Weather"]
    )

    # Convert Temperature into numerical values
    data["Temperature"] = temperature_encoder.fit_transform(
        data["Temperature"]
    )

    # Convert Play into numerical values
    data["Play"] = play_encoder.fit_transform(
        data["Play"]
    )

    print("Encoded Dataset:")
    print(data)
    print()


    # --------------------------------------------------------
    # Separate Features and Target
    # --------------------------------------------------------

    X = data[["Weather", "Temperature"]]
    Y = data["Play"]


    # --------------------------------------------------------
    # Step 3: Train Data
    # --------------------------------------------------------

    print("Step 3: Train Data")
    print("-----------------")

    # Create KNN classifier with K = 3
    model = KNeighborsClassifier(n_neighbors=3)

    # Train the model using the complete dataset
    model.fit(X, Y)

    print("Model trained successfully.")
    print()


    # --------------------------------------------------------
    # Step 4: Test Data
    # --------------------------------------------------------

    print("Step 4: Test Data")
    print("----------------")

    # Test case:
    # Weather    = Sunny
    # Temperature = Cool

    weather = "Sunny"
    temperature = "Cool"

    # Convert test values using the same encoders
    weather_value = weather_encoder.transform([weather])[0]
    temperature_value = temperature_encoder.transform(
        [temperature]
    )[0]

    # Create test data
    test_data = [[weather_value, temperature_value]]

    # Predict result
    prediction = model.predict(test_data)

    # Convert numerical prediction back to Yes/No
    result = play_encoder.inverse_transform(prediction)

    print("Weather     :", weather)
    print("Temperature :", temperature)
    print("Prediction  :", result[0])
    print()


    # --------------------------------------------------------
    # Step 5: Calculate Accuracy
    # --------------------------------------------------------

    print("Step 5: Calculate Accuracy")
    print("-------------------------")

    # Split dataset into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.5,
        random_state=42
    )

    # Train model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, Y_train)

    # Predict testing data
    Y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)

    print("Accuracy for K = 3 :", accuracy * 100, "%")
    print()


    # --------------------------------------------------------
    # Calculate Accuracy by Changing Value of K
    # --------------------------------------------------------

    print("Accuracy by changing value of K:")
    print("--------------------------------")

    for k in range(1, 6):

        accuracy = CheckAccuracy(X, Y, k)

        print(
            "K =",
            k,
            "Accuracy =",
            round(accuracy * 100, 2),
            "%"
        )


# ------------------------------------------------------------
# Program Execution
# ------------------------------------------------------------
if __name__ == "__main__":
    main()