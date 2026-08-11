# Problem Statement:
# Use value_counts() to analyze the distribution of FinalResult.
# Calculate the percentage of Pass and Fail students.
# Determine whether the dataset is balanced.

import pandas as pd


def main():
    # Load the dataset
    df = pd.read_csv("student_performance_ml.csv")

    # Count Pass and Fail students
    result_counts = df["FinalResult"].value_counts()

    print("Distribution of FinalResult:")
    print(result_counts)

    # Calculate percentages
    result_percentages = df["FinalResult"].value_counts(normalize=True) * 100

    print("\nPercentage of Students:")
    print(result_percentages)

    # Get Pass and Fail percentages
    pass_percentage = result_percentages.get(1, 0)
    fail_percentage = result_percentages.get(0, 0)

    print("\nPass Percentage:", round(pass_percentage, 2), "%")
    print("Fail Percentage:", round(fail_percentage, 2), "%")

    # Check whether dataset is balanced
    difference = abs(pass_percentage - fail_percentage)

    if difference <= 10:
        print("\nThe dataset is approximately balanced.")
    else:
        print("\nThe dataset is not balanced.")


if __name__ == "__main__":
    main()