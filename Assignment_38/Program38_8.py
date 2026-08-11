# Problem Statement:
# Draw a boxplot for Attendance.
# Identify if any outliers are present.

import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Load the dataset
    df = pd.read_csv("student_performance_ml.csv")

    # Draw boxplot
    plt.boxplot(df["Attendance"])

    plt.title("Boxplot of Attendance")
    plt.ylabel("Attendance (%)")
    plt.grid(axis="y", alpha=0.3)

    plt.show()

    # Calculate Q1 and Q3
    Q1 = df["Attendance"].quantile(0.25)
    Q3 = df["Attendance"].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Calculate lower and upper limits
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Find outliers
    outliers = df[
        (df["Attendance"] < lower_limit) |
        (df["Attendance"] > upper_limit)
    ]

    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)

    if len(outliers) > 0:
        print("\nOutliers are present:")
        print(outliers["Attendance"])
    else:
        print("\nNo outliers are present in Attendance.")


if __name__ == "__main__":
    main()