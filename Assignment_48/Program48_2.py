# Problem Statement:
# Using the same dataset from the above question, calculate model performance.
#
# Dataset:
# X = [1, 2, 3, 4, 5]
# Y = [3, 4, 2, 4, 5]
#
# Tasks:
# 1. Predict all Y values using the regression equation.
# 2. Calculate Mean Squared Error (MSE).
# 3. Calculate R² Score.
# 4. Show all intermediate calculations.


def main():

    X = [1, 2, 3, 4, 5]
    Y = [3, 4, 2, 4, 5]

    # Regression equation obtained from Question 1
    m = 0.4
    c = 2.4

    # Calculate predicted Y values
    predicted_y = []

    for x in X:
        y_pred = m * x + c
        predicted_y.append(y_pred)

    print("Regression Equation:")
    print("Y =", m, "X +", c)

    print("\nActual Y values:")
    print(Y)

    print("\nPredicted Y values:")
    print(predicted_y)

    # Calculate squared errors
    squared_errors = []

    print("\nIntermediate Calculations:")
    print("--------------------------------")

    for i in range(len(Y)):
        error = Y[i] - predicted_y[i]
        squared_error = error ** 2

        squared_errors.append(squared_error)

        print("X =", X[i])
        print("Actual Y =", Y[i])
        print("Predicted Y =", predicted_y[i])
        print("Error =", error)
        print("Squared Error =", squared_error)
        print()

    # Calculate MSE
    mse = sum(squared_errors) / len(squared_errors)

    print("Sum of Squared Errors =", sum(squared_errors))
    print("MSE =", mse)

    # Calculate R² Score
    mean_y = sum(Y) / len(Y)

    total_sum_of_squares = 0

    for y in Y:
        total_sum_of_squares += (y - mean_y) ** 2

    r2 = 1 - (sum(squared_errors) / total_sum_of_squares)

    print("\nMean of Y =", mean_y)
    print("Total Sum of Squares =", total_sum_of_squares)
    print("R² Score =", r2)


if __name__ == "__main__":
    main()