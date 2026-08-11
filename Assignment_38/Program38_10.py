# Problem Statement:
# Plot SleepHours against FinalResult.
# Determine whether sleeping more guarantees success.
# Explain your observation.

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
        passed["FinalResult"],
        passed["SleepHours"],
        label="Pass"
    )

    # Plot failed students
    plt.scatter(
        failed["FinalResult"],
        failed["SleepHours"],
        label="Fail"
    )

    plt.title("SleepHours vs FinalResult")
    plt.xlabel("Final Result (0 = Fail, 1 = Pass)")
    plt.ylabel("Sleep Hours")
    plt.xticks([0, 1], ["Fail", "Pass"])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

    # Calculate average sleep hours
    average_sleep = df.groupby("FinalResult")["SleepHours"].mean()

    print("Average Sleep Hours:")
    print(average_sleep)

    print("\nObservation:")

    if average_sleep[1] > average_sleep[0]:
        print("Passed students sleep more on average than failed students.")
    else:
        print("Passed students do not sleep more on average than failed students.")

    print("Sleeping more does not guarantee success.")
    print("FinalResult can depend on several factors such as StudyHours, Attendance,")
    print("PreviousScore and AssignmentsCompleted.")


if __name__ == "__main__":
    main()