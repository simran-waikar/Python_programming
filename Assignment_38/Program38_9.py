# Problem Statement:
# Create a plot showing the relationship between AssignmentsCompleted
# and FinalResult.
# Explain your observation.

import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Load the dataset
    df = pd.read_csv("student_performance_ml.csv")

    # Calculate average assignments completed for each result
    assignment_data = df.groupby("FinalResult")["AssignmentsCompleted"].mean()

    # Plot bar chart
    assignment_data.plot(kind="bar")

    plt.title("Assignments Completed vs Final Result")
    plt.xlabel("Final Result (0 = Fail, 1 = Pass)")
    plt.ylabel("Average Assignments Completed")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)

    plt.show()

    # Display values
    print("Average Assignments Completed:")
    print(assignment_data)

    print("\nObservation:")

    if assignment_data[1] > assignment_data[0]:
        print("Students who passed completed more assignments on average.")
        print("This suggests that completing assignments may be associated with better performance.")
    else:
        print("Students who passed did not complete more assignments on average.")
        print("Therefore, assignments completed may not have a strong relationship with the result.")


if __name__ == "__main__":
    main()