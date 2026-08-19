# Problem Statement:
# Implement Simple Linear Regression manually without using any ML library.
#
# Dataset:
# X = [1, 2, 3, 4, 5]
# Y = [3, 4, 2, 4, 5]
#
# Calculate:
# 1. Mean of X
# 2. Mean of Y
# 3. Slope (m)
# 4. Intercept (c)
# 5. Regression equation
# 6. Predicted Y for X = 6


def main():

    X = [1, 2, 3, 4, 5]
    Y = [3, 4, 2, 4, 5]

    # Calculate mean of X and Y
    mean_x = sum(X) / len(X)
    mean_y = sum(Y) / len(Y)

    # Calculate slope
    numerator = 0
    denominator = 0

    for i in range(len(X)):
        numerator += (X[i] - mean_x) * (Y[i] - mean_y)
        denominator += (X[i] - mean_x) ** 2

    m = numerator / denominator

    # Calculate intercept
    c = mean_y - (m * mean_x)

    # Regression equation
    predicted_y = m * 6 + c

    print("Mean of X =", mean_x)
    print("Mean of Y =", mean_y)

    print("\nSlope (m) =", m)
    print("Intercept (c) =", c)

    print("\nRegression Equation:")
    print("Y =", m, "X +", c)

    print("\nPredicted Y for X = 6:", predicted_y)


if __name__ == "__main__":
    main()