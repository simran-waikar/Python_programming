# Problem Statement:
# Consider the following data:
#
# Actual Values:
# [1, 1, 1, 0, 0, 0, 0, 0]
#
# Predicted Values:
# [1, 1, 0, 1, 0, 1, 0, 0]
#
# Determine:
# True Positive (TP)
# True Negative (TN)
# False Positive (FP)
# False Negative (FN)


def main():
    actual = [1, 1, 1, 0, 0, 0, 0, 0]
    predicted = [1, 1, 0, 1, 0, 1, 0, 0]

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for actual_value, predicted_value in zip(actual, predicted):

        if actual_value == 1 and predicted_value == 1:
            tp += 1

        elif actual_value == 0 and predicted_value == 0:
            tn += 1

        elif actual_value == 0 and predicted_value == 1:
            fp += 1

        elif actual_value == 1 and predicted_value == 0:
            fn += 1

    print("True Positive (TP):", tp)
    print("True Negative (TN):", tn)
    print("False Positive (FP):", fp)
    print("False Negative (FN):", fn)


if __name__ == "__main__":
    main()