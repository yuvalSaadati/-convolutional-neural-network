import pandas as pd


def load_data(filename):
    """
    Loads data from a CSV file and processes it into suitable format.
    For training and validation data, it splits into features and labels.
    For test data, it returns only features and assumes the first column is a placeholder.
    """
    data = pd.read_csv(filename, header=None)
    labels = data.iloc[:, 0].values - 1  # Convert class labels to zero-indexed
    features = data.iloc[:, 1:].values
    # else:
    #     # Test data does not include labels
    #     labels = None
    #     features = data.iloc[:, 1:].values  # Ignore the placeholder column

    # Reshape features into 32x32x3 format for RGB images
    features = features.reshape((-1, 3, 32, 32))
    return features, labels