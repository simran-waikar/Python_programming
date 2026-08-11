# Problem Statement:
# Create a scatter plot of StudyHours vs PreviousScore.
# Use different colors for Pass and Fail students.

import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Load the dataset
    df = pd.read_csv("student_performance_ml.csv")

    # Separate passed and failed students
    passed = df[df["FinalResult"] == 1]
    failed = df[df["FinalResult"] == 0]

    # Plot passed students
    plt.scatter(
        passed["StudyHours"],
        passed["PreviousScore"],
        label="Pass"
    )

    # Plot failed students
    plt.scatter(
        failed["StudyHours"],
        failed["PreviousScore"],
        label="Fail"
    )

    plt.title("StudyHours vs PreviousScore")
    plt.xlabel("Study Hours")
    plt.ylabel("Previous Score")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()