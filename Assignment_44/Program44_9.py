# Problem Statement:
# Create a DataFrame with missing values and fill them with
# the respective column mean.

import pandas as pd
import numpy as np


def main():
    data2 = {
        'Name': ['Amit', 'Sagar', 'Pooja'],
        'Math': [np.nan, 76, 88],
        'Science': [91, np.nan, 85]
    }

    df = pd.DataFrame(data2)

    print("DataFrame before filling missing values:")
    print(df)

    numeric_columns = ['Math', 'Science']

    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].mean())

    print("\nDataFrame after filling missing values:")
    print(df)


if __name__ == "__main__":
    main()