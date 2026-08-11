# Problem Statement 8:
# Write a single structured Python program that performs:
# 1. Dataset loading
# 2. Data analysis
# 3. Visualization
# 4. Train-test split
# 5. Model training
# 6. Prediction
# 7. Accuracy calculation
# 8. Confusion matrix generation
# 9. Final conclusion


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def main():

    # ---------------------------------------------------------
    # STEP 1: DATASET LOADING
    # ---------------------------------------------------------

    print("STEP 1: Loading Dataset")
    print("-----------------------")

    data = pd.read_csv("student_performance_ml.csv")

    print("Dataset loaded successfully.")
    print("Number of rows:", data.shape[0])
    print("Number of columns:", data.shape[1])


    # ---------------------------------------------------------
    # STEP 2: DATA ANALYSIS
    # ---------------------------------------------------------

    print("\nSTEP 2: Data Analysis")
    print("--------------------")

    print("\nFirst five records:")
    print(data.head())

    print("\nDataset information:")
    print(data.info())

    print("\nStatistical summary:")
    print(data.describe())

    print("\nMissing values:")
    print(data.isnull().sum())


    # ---------------------------------------------------------
    # STEP 3: VISUALIZATION
    # ---------------------------------------------------------

    print("\nSTEP 3: Data Visualization")
    print("-------------------------")

    # Histogram of Study Hours
    plt.figure(figsize=(8, 5))
    plt.hist(data["StudyHours"], bins=10)
    plt.xlabel("Study Hours")
    plt.ylabel("Number of Students")
    plt.title("Distribution of Study Hours")
    plt.show()

    # Scatter plot of Study Hours vs Previous Score
    plt.figure(figsize=(8, 5))
    plt.scatter(data["StudyHours"], data["PreviousScore"])
    plt.xlabel("Study Hours")
    plt.ylabel("Previous Score")
    plt.title("Study Hours vs Previous Score")
    plt.show()

    # Final Result distribution
    plt.figure(figsize=(6, 4))
    data["FinalResult"].value_counts().plot(kind="bar")
    plt.xlabel("Final Result")
    plt.ylabel("Number of Students")
    plt.title("Pass/Fail Distribution")
    plt.xticks(rotation=0)
    plt.show()


    # ---------------------------------------------------------
    # STEP 4: TRAIN-TEST SPLIT
    # ---------------------------------------------------------

    print("\nSTEP 4: Train-Test Split")
    print("-----------------------")

    # Input features
    X = data[[
        "StudyHours",
        "Attendance",
        "PreviousScore",
        "AssignmentsCompleted",
        "SleepHours"
    ]]

    # Target variable
    y = data["FinalResult"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))


    # ---------------------------------------------------------
    # STEP 5: MODEL TRAINING
    # ---------------------------------------------------------

    print("\nSTEP 5: Model Training")
    print("---------------------")

    # Create Decision Tree model
    model = DecisionTreeClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    print("Decision Tree model trained successfully.")


    # ---------------------------------------------------------
    # STEP 6: PREDICTION
    # ---------------------------------------------------------

    print("\nSTEP 6: Prediction")
    print("-----------------")

    # Predict test data
    y_pred = model.predict(X_test)

    print("Actual Values    Predicted Values")
    print("--------------------------------")

    for actual, predicted in zip(y_test, y_pred):
        print(f"{actual:<17} {predicted}")


    # ---------------------------------------------------------
    # STEP 7: ACCURACY CALCULATION
    # ---------------------------------------------------------

    print("\nSTEP 7: Accuracy Calculation")
    print("---------------------------")

    # Training prediction
    train_prediction = model.predict(X_train)

    # Calculate training accuracy
    training_accuracy = accuracy_score(
        y_train,
        train_prediction
    )

    # Calculate testing accuracy
    testing_accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("Training Accuracy:", training_accuracy * 100, "%")
    print("Testing Accuracy :", testing_accuracy * 100, "%")


    # ---------------------------------------------------------
    # STEP 8: CONFUSION MATRIX
    # ---------------------------------------------------------

    print("\nSTEP 8: Confusion Matrix")
    print("-----------------------")

    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    # Display confusion matrix
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fail", "Pass"]
    )

    display.plot()
    plt.title("Decision Tree Confusion Matrix")
    plt.show()


    # ---------------------------------------------------------
    # STEP 9: FINAL CONCLUSION
    # ---------------------------------------------------------

    print("\nSTEP 9: Final Conclusion")
    print("-----------------------")

    print("Training Accuracy:", round(training_accuracy * 100, 2), "%")
    print("Testing Accuracy :", round(testing_accuracy * 100, 2), "%")

    difference = training_accuracy - testing_accuracy

    if difference > 0.15:
        print("\nConclusion: The model is likely OVERFITTING.")
        print("The training accuracy is considerably higher than")
        print("the testing accuracy.")

    elif training_accuracy < 0.70 and testing_accuracy < 0.70:
        print("\nConclusion: The model may be UNDERFITTING.")
        print("Both training and testing accuracies are relatively low.")

    else:
        print("\nConclusion: The model has a reasonable fit.")
        print("Training and testing accuracies are relatively close.")

    print("\nMachine Learning analysis completed successfully.")


# -------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------

if __name__ == "__main__":
    main()