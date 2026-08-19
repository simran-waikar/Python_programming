# Problem Statement:
# Euclidean distance before and after feature scaling
#
# Dataset:
# [[25, 20000],
#  [30, 40000],
#  [35, 80000]]


import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    data = np.array([
        [25, 20000],
        [30, 40000],
        [35, 80000]
    ])

    # Select two points
    point1 = data[0]
    point2 = data[1]

    # Calculate Euclidean distance before scaling
    distance_before = np.linalg.norm(point1 - point2)

    # Apply feature scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Select the corresponding scaled points
    scaled_point1 = scaled_data[0]
    scaled_point2 = scaled_data[1]

    # Calculate Euclidean distance after scaling
    distance_after = np.linalg.norm(scaled_point1 - scaled_point2)

    print("Point 1:", point1)
    print("Point 2:", point2)

    print("\nEuclidean Distance Before Scaling:", distance_before)

    print("\nScaled Point 1:", scaled_point1)
    print("Scaled Point 2:", scaled_point2)

    print("\nEuclidean Distance After Scaling:", distance_after)

    print("\nExplanation:")
    print("Before scaling, the second feature has much larger values")
    print("and therefore has a greater influence on the distance.")
    print("After scaling, both features are brought to a similar scale,")
    print("so neither feature dominates the Euclidean distance.")


if __name__ == "__main__":
    main()