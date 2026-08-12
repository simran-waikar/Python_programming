# Problem Statement:
# Write a Python program that classifies a new data point using the
# K-Nearest Neighbors (KNN) algorithm.
# The algorithm must be implemented manually without using any
# machine learning library.
#
# Tasks:
# 1. Accept X and Y coordinates of a new point.
# 2. Calculate Euclidean distance from all dataset points.
# 3. Sort the distances.
# 4. Select K = 3 nearest neighbors.
# 5. Predict the class using majority voting.


import math


def main():

    # Dataset
    data = [
        ("A", 1, 2, "Red"),
        ("B", 2, 3, "Red"),
        ("C", 3, 1, "Blue"),
        ("D", 6, 5, "Blue")
    ]

    # Accept new point from user
    x = float(input("Enter X coordinate: "))
    y = float(input("Enter Y coordinate: "))

    # Calculate Euclidean distances
    distances = []

    for point, px, py, label in data:
        distance = math.sqrt((x - px) ** 2 + (y - py) ** 2)
        distances.append((distance, point, label))

    # Sort distances
    distances.sort()

    # Select K = 3 nearest neighbors
    k = 3
    nearest_neighbors = distances[:k]

    # Display nearest neighbors
    print("\nNearest Neighbors:")

    for distance, point, label in nearest_neighbors:
        print(f"{point} - Distance: {distance:.2f}")

    # Majority voting
    votes = {}

    for distance, point, label in nearest_neighbors:
        votes[label] = votes.get(label, 0) + 1

    predicted_class = max(votes, key=votes.get)

    print("\nPredicted Class:", predicted_class)


if __name__ == "__main__":
    main()