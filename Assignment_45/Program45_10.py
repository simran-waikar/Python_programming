# Problem Statement:
# Plot a boxplot for English marks to check distribution and outliers.

import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("students.csv")

    plt.boxplot(df["English"])

    plt.ylabel("English Marks")
    plt.title("Boxplot of English Marks")

    plt.show()


if __name__ == "__main__":
    main()