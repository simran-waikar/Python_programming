# Problem Statement:
# Use K-Nearest Neighbors (KNN) to predict whether a student
# passes or fails based on study hours and attendance.
#
# Dataset:
# Study Hours    Attendance    Result
#     2              60         Fail
#     5              80         Pass
#     6              85         Pass
#     1              50         Fail
#
# Tasks:
# 1. Accept study hours from the user.
# 2. Accept attendance percentage from the user.
# 3. Apply the KNN algorithm.
# 4. Predict whether the student Passes or Fails.


import math


def main():

    # Dataset
    data = [
        (2, 60, "Fail"),
        (5, 80, "Pass"),
        (6, 85, "Pass"),
        (1, 50, "Fail")
    ]

    # Accept input from user
    study_hours = float(input("Enter Study Hours: "))
    attendance = float(input("Enter Attendance: "))

    # Calculate Euclidean distances
    distances = []

    for hours, attend, result in data:

        distance = math.sqrt(
            (study_hours - hours) ** 2 +
            (attendance - attend) ** 2
        )

        distances.append((distance, result))

    # Sort distances
    distances.sort()

    # Select K nearest neighbors
    k = 3
    nearest_neighbors = distances[:k]

    # Majority voting
    votes = {}

    for distance, result in nearest_neighbors:
        votes[result] = votes.get(result, 0) + 1

    # Predict result
    predicted_result = max(votes, key=votes.get)

    print("\nPredicted Result:", predicted_result)


if __name__ == "__main__":
    main()