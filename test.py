import numpy as np

from utils import predict


def test_network(x, y, layers):
    predictions = calc_predictions(x, layers)
    score = accuracy_score(y, predictions, True)
    print(f'accuracy score is: {score:2.f}%')


def calc_predictions(x, layers):
    predictions = []

    for data in x:
        pred = predict(data, layers)
        one_hot_pred = np.zeros_like(pred)
        one_hot_pred[np.argmax(pred)] = 1
        predictions.append(one_hot_pred.flatten())

    return np.array(predictions)


def accuracy_score(y_true, y_pred, normalize=True):
    """
    Computes the accuracy classification score.

    Parameters:
        y_true (array-like): Ground truth (correct) labels.
        y_pred (array-like): Predicted labels.
        normalize (bool, optional): If True, return the fraction of correctly classified samples.
                                    If False, return the number of correctly classified samples.

    Returns:
        float or int: Accuracy score. If `normalize=True`, returns accuracy as a float (0 to 1).
                      If `normalize=False`, returns the number of correct predictions.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure the inputs have the same length
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    correct_predictions = np.sum(y_true == y_pred)  # Count correct predictions

    if normalize:
        return correct_predictions / len(y_true) * 100.0  # Return accuracy as a fraction
    else:
        return correct_predictions  # Return the count of correct predictions
