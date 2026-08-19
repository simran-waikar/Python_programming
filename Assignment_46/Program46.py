# Problem Statement:
# Design a Machine Learning application using Linear Regression


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main():

    # Step 1: Get Data
    # Load the dataset
    data = pd.read_csv("Advertising.csv")

    print("Dataset:")
    print(data)

    # Step 2: Clean, Prepare and Manipulate Data

    # Select input features
    X = data[["TV", "radio", "newspaper"]]

    # Select output/target feature
    Y = data["sales"]

    print("\nInput Features:")
    print(X)

    print("\nTarget Feature:")
    print(Y)

    # Step 3: Train the Data

    # Divide dataset into training and testing data
    # 80% data is used for training and 20% for testing
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print("\nTraining Data:")
    print(X_train)

    print("\nTesting Data:")
    print(X_test)

    # Create Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, Y_train)

    # Step 4: Test the Data

    # Predict sales using testing data
    Y_pred = model.predict(X_test)

    # Step 5: Display predicted and expected values

    print("\nPredicted Sales vs Expected Sales:")

    for predicted, expected in zip(Y_pred, Y_test):
        print("Predicted:", round(predicted, 2),
              "Expected:", expected)

    # Display model coefficients
    print("\nModel Coefficients:")
    print("TV:", model.coef_[0])
    print("Radio:", model.coef_[1])
    print("Newspaper:", model.coef_[2])

    print("\nModel Intercept:")
    print(model.intercept_)


if __name__ == "__main__":
    main()