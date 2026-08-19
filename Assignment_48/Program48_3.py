# Problem Statement:
# Consider the following dataset:
#
# Experience    Salary
# 1             20000
# 2             25000
# 3             30000
# 4             35000
# 5             40000
#
# Tasks:
# 1. Train a Linear Regression model.
# 2. Predict salary for 6 years of experience.
# 3. Plot regression line using matplotlib.
#
# Graph should display:
# - Data points
# - Regression line


from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def main():

    # Dataset
    X = [[1], [2], [3], [4], [5]]
    Y = [20000, 25000, 30000, 35000, 40000]

    # Create Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X, Y)

    # Predict salary for 6 years of experience
    experience = [[6]]
    predicted_salary = model.predict(experience)

    print("Slope =", model.coef_[0])
    print("Intercept =", model.intercept_)

    print("\nPredicted Salary for 6 Years Experience: ₹",
          int(predicted_salary[0]))

    # Predict values for regression line
    predicted_values = model.predict(X)

    # Plot data points
    plt.scatter(X, Y, label="Data Points")

    # Plot regression line
    plt.plot(X, predicted_values, label="Regression Line")

    # Labels and title
    plt.xlabel("Experience (Years)")
    plt.ylabel("Salary")
    plt.title("Experience vs Salary - Linear Regression")

    plt.legend()
    plt.grid(True)

    # Display graph
    plt.show()


if __name__ == "__main__":
    main()