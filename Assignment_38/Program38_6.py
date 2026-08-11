# Problem Statement:
# Plot a histogram of StudyHours.
# Explain what the distribution tells you.

import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Load the dataset
    df = pd.read_csv("student_performance_ml.csv")

    # Plot histogram
    plt.hist(df["StudyHours"], bins=10, edgecolor="black")

    plt.title("Distribution of Study Hours")
    plt.xlabel("Study Hours per Day")
    plt.ylabel("Number of Students")
    plt.grid(axis="y", alpha=0.3)

    plt.show()

    # Display basic observations
    print("Observation:")
    print("The histogram shows how StudyHours are distributed among students.")
    print("The tallest bars represent the study-hour range containing the most students.")
    print("It also helps identify whether the data is concentrated or spread across different study hours.")


if __name__ == "__main__":
    main()