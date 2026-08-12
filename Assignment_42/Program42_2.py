# Problem Statement:
# Write a Python program that demonstrates how prediction changes
# when the value of K changes in the K-Nearest Neighbors algorithm.
#
# Use the same dataset as Assignment 1.
#
# Predict the class of the same new point using:
# K = 1
# K = 3
# K = 5
#
# Also display the prediction results.


import math


def knn_predict(data, x, y, k):

    # Calculate distances
    distances = []

    for point, px, py, label in data:
        distance = math.sqrt((x - px) ** 2 + (y - py) ** 2)
        distances.append((distance, label))

    # Sort distances
    distances.sort()

    # K cannot be greater than the number of data points
    k = min(k, len(data))

    # Select K nearest neighbors
    nearest_neighbors = distances[:k]

    # Majority voting
    votes = {}

    for distance, label in nearest_neighbors:
        votes[label] = votes.get(label, 0) + 1

    # Find majority class
    predicted_class = max(votes, key=votes.get)

    return predicted_class


def main():

    # Dataset
    data = [
        ("A", 1, 2, "Red"),
        ("B", 2, 3, "Red"),
        ("C", 3, 1, "Blue"),
        ("D", 6, 5, "Blue")
    ]

    # Same new point as Assignment 1
    x = float(input("Enter X coordinate: "))
    y = float(input("Enter Y coordinate: "))

    print("\nPrediction Results")

    # K = 1
    result1 = knn_predict(data, x, y, 1)
    print("K = 1 ->", result1)

    # K = 3
    result3 = knn_predict(data, x, y, 3)
    print("K = 3 ->", result3)

    # K = 5
    result5 = knn_predict(data, x, y, 5)
    print("K = 5 ->", result5)

    print("\nExplanation:")
    print("As K increases, more neighboring data points are considered.")
    print("Therefore, distant points can influence the majority vote.")
    print("This can sometimes change the predicted class.")


if __name__ == "__main__":
    main()