# Problem Statement:
# Plot a pie chart of subject marks for 'Sagar'.

import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("students.csv")

    student = df[df["Name"] == "Sagar"]

    if student.empty:
        print("Student 'Sagar' not found.")
        return

    marks = [
        student["Math"].iloc[0],
        student["English"].iloc[0],
        student["Science"].iloc[0]
    ]

    subjects = ["Math", "English", "Science"]

    plt.pie(marks, labels=subjects, autopct="%1.1f%%")
    plt.title("Subject Marks of Sagar")
    plt.show()


if __name__ == "__main__":
    main()