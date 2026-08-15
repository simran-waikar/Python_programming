# Problem Statement:
# Export the final DataFrame to a CSV file.

import pandas as pd


def main():
    df = pd.read_csv("students.csv")

    # Calculate total marks
    df["Total"] = df[["Math", "English", "Science"]].sum(axis=1)

    # Add Status column
    df["Status"] = df["Total"].apply(
        lambda total: "Pass" if total >= 250 else "Fail"
    )

    # Export DataFrame to CSV
    df.to_csv("final_students.csv", index=False)

    print("Final DataFrame exported successfully to final_students.csv")


if __name__ == "__main__":
    main()