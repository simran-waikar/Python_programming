# ============================================================
# PROBLEM STATEMENT:
# Breast Cancer Prediction
#
# Develop a machine learning classification model using the
# Breast Cancer Wisconsin Dataset to predict whether a tumor
# is Malignant (0) or Benign (1).
#
# ============================================================


# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def main():

    # --------------------------------------------------------
    # 1. LOAD THE DATASET
    # --------------------------------------------------------

    print("=" * 60)
    print("BREAST CANCER PREDICTION")
    print("=" * 60)

    # Load CSV file
    df = pd.read_csv("breast-cancer-wisconsin.csv")

    print("\nDataset loaded successfully.")

    # Display first 5 records
    print("\nFirst 5 Records:")
    print(df.head())

    # Display number of records and features
    print("\nNumber of Records:", df.shape[0])
    print("Number of Columns:", df.shape[1])

    # --------------------------------------------------------
    # 2. EXPLORE THE DATASET
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print("\nColumn Names:")
    print(df.columns)

    print("\nDataset Information:")
    df.info()

    # --------------------------------------------------------
    # 3. REMOVE UNNECESSARY COLUMNS
    # --------------------------------------------------------

    # The standard dataset contains an 'id' column which is
    # only an identifier and should not be used for prediction.

    if "id" in df.columns:
        df = df.drop("id", axis=1)

    # Remove unnamed index column if present
    if "Unnamed: 32" in df.columns:
        df = df.drop("Unnamed: 32", axis=1)

    # --------------------------------------------------------
    # 4. HANDLE MISSING VALUES
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    print(df.isnull().sum())

    # Convert empty strings to NaN
    df = df.replace("", pd.NA)

    # Fill missing numerical values with column mean
    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].mean()
    )

    print("\nMissing values handled successfully.")

    # --------------------------------------------------------
    # 5. CONVERT TARGET VARIABLE
    # --------------------------------------------------------

    # diagnosis:
    # M = Malignant
    # B = Benign
    #
    # Required target representation:
    # 0 = Malignant
    # 1 = Benign

    if "diagnosis" in df.columns:

        df["diagnosis"] = df["diagnosis"].map({
            "M": 0,
            "B": 1
        })

    else:
        print("\nError: 'diagnosis' column not found.")
        return

    print("\nTarget Variable:")
    print("0 -> Malignant")
    print("1 -> Benign")

    # --------------------------------------------------------
    # 6. SEPARATE FEATURES AND TARGET
    # --------------------------------------------------------

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    print("\nNumber of Features:", X.shape[1])

    # --------------------------------------------------------
    # 7. TARGET DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION")
    print("=" * 60)

    print(y.value_counts())

    # Visualize target distribution
    plt.figure(figsize=(7, 5))

    sns.countplot(x=y)

    plt.title("Distribution of Tumor Types")
    plt.xlabel("Tumor Type")
    plt.ylabel("Number of Records")

    plt.xticks(
        ticks=[0, 1],
        labels=["Malignant", "Benign"]
    )

    plt.show()

    # --------------------------------------------------------
    # 8. SUMMARY STATISTICS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(X.describe())

    # --------------------------------------------------------
    # 9. FEATURE CORRELATION
    # --------------------------------------------------------

    plt.figure(figsize=(15, 12))

    correlation_matrix = X.corr()

    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        linewidths=0.1
    )

    plt.title("Feature Correlation Heatmap")
    plt.show()

    # --------------------------------------------------------
    # 10. TRAIN-TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLIT")
    print("=" * 60)

    print("Training Records:", X_train.shape[0])
    print("Testing Records:", X_test.shape[0])

    # --------------------------------------------------------
    # 11. FEATURE SCALING
    # --------------------------------------------------------

    scaler = StandardScaler()

    # Fit only on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform testing data
    X_test_scaled = scaler.transform(X_test)

    print("\nFeature scaling completed successfully.")

    # --------------------------------------------------------
    # 12. BUILD THE MACHINE LEARNING MODEL
    # --------------------------------------------------------

    model = LogisticRegression(max_iter=5000)

    # Train the model
    model.fit(X_train_scaled, y_train)

    print("\nLogistic Regression model trained successfully.")

    # --------------------------------------------------------
    # 13. MAKE PREDICTIONS
    # --------------------------------------------------------

    y_pred = model.predict(X_test_scaled)

    print("\nPredictions generated successfully.")

    # --------------------------------------------------------
    # 14. CALCULATE ACCURACY
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print("\nAccuracy:", round(accuracy * 100, 2), "%")

    # --------------------------------------------------------
    # 15. CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # Display confusion matrix
    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"]
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

    # --------------------------------------------------------
    # 16. PRECISION, RECALL AND F1-SCORE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Malignant", "Benign"]
        )
    )

    # --------------------------------------------------------
    # 17. DISPLAY INDIVIDUAL METRICS
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)

    print("\nMalignant:")
    print("Precision:", round(report["0"]["precision"], 4))
    print("Recall   :", round(report["0"]["recall"], 4))
    print("F1-Score :", round(report["0"]["f1-score"], 4))

    print("\nBenign:")
    print("Precision:", round(report["1"]["precision"], 4))
    print("Recall   :", round(report["1"]["recall"], 4))
    print("F1-Score :", round(report["1"]["f1-score"], 4))

    # --------------------------------------------------------
    # 18. OBSERVATIONS AND CONCLUSION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("OBSERVATIONS AND CONCLUSION")
    print("=" * 60)

    print("""
Observations:

1. The Breast Cancer Wisconsin dataset contains medical
   features extracted from breast cancer biopsy images.

2. The target variable contains two classes:
   0 -> Malignant
   1 -> Benign

3. Missing values were checked and handled.

4. The features were standardized using StandardScaler.

5. The dataset was divided into 80% training data and
   20% testing data.

6. Logistic Regression was used as the classification model.

7. The model was evaluated using Accuracy, Confusion Matrix,
   Precision, Recall and F1-Score.

Conclusion:

The Logistic Regression model can classify breast tumors as
Malignant or Benign based on the given medical features.
The evaluation metrics help determine the effectiveness of
the classification model.
""")


# ------------------------------------------------------------
# PROGRAM ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    main()