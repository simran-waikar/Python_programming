# Problem Statement:
# Plot a histogram of math marks.

import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("students.csv")

    plt.hist(df["Math"], bins=5, edgecolor="black")

    plt.xlabel("Math Marks")
    plt.ylabel("Number of Students")
    plt.title("Histogram of Math Marks")

    plt.show()


if __name__ == "__main__":
    main()