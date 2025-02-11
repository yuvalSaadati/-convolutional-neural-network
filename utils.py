import random

import numpy as np

random.seed(3)



def cross_entropy_loss(predictions, targets):
    epsilon = 1e-7  # To avoid log(0)
    # batch_size = predictions.shape[0]
    batch_size, num_samples = predictions.shape
    # Clip predictions to avoid numerical issues with log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    # Compute the loss per sample, then take the mean
    loss = -np.sum(targets * np.log(predictions)) / (batch_size )
    return loss

def cross_entropy_loss_gradient(actual_labels, predicted_probs):
    epsilon = 1e-7  # To avoid division by zero
    # batch_size = actual_labels.shape[0]
    batch_size, num_samples = actual_labels.shape

    gradient = -actual_labels / (predicted_probs + epsilon) / (batch_size )
    return gradient


def predict(input_sample, layers):
    # Forward pass through Convolution and pooling
    conv_out = layers[0].forward(input_sample)
    pool_out = layers[1].forward(conv_out)
    conv_out = layers[2].forward(pool_out)
    pool_out = layers[3].forward(conv_out)
    # Flattening
    flattened_output = pool_out.flatten()
    # Forward pass through fully connected layer
    predictions = layers[4].forward(flattened_output)
    return predictions


def softmax_derivative(s):
    return np.diagflat(s) - np.dot(s, s.T)


def softmax(z):
    # Shift the input values to avoid numerical instability
    shifted_z = z - np.max(z, axis=1, keepdims=True)
    exp_values = np.exp(shifted_z)
    sum_exp_values = np.sum(exp_values, axis=1, keepdims=True)
    # log_sum_exp = np.log(sum_exp_values)

    # Compute the softmax probabilities
    probabilities = exp_values / sum_exp_values

    return probabilities


def to_categorical(y, num_classes=10, dtype="float32"):
    """
    Converts a class vector (integers) to a binary class matrix (one-hot encoding).

    Parameters:
        y (array-like): Class vector to be converted into a one-hot matrix.
        num_classes (int, optional): Total number of classes. If not provided, it is inferred.
        dtype (str, optional): Data type of the returned matrix.

    Returns:
        numpy.ndarray: A one-hot encoded matrix representation of `y`.
    """
    y = np.array(y, dtype="int")  # Ensure it's an integer array
    if num_classes is None:
        num_classes = np.max(y) + 1  # Infer number of classes if not given

    # Create a zero matrix of shape (num_samples, num_classes)
    categorical = np.zeros((y.shape[0], num_classes), dtype=dtype)

    # Set the corresponding indices to 1
    categorical[np.arange(y.shape[0]), y] = 1

    return categorical


def augment_data(x):
    augmented_x = np.empty(len(x))
    i = 0
    for img in x:
        if random.random() > 0.5:
            img[0] = np.fliplr(img[0])
        if random.random() > 0.5:
            img = np.rot90(img, k=random.choice([1, 2, 3]))
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.02, img.shape)
            img = np.clip(img + noise, 0, 1)
        if random.random() > 0.5:
            img = img * (0.8 + 0.4 * np.random.rand())
        augmented_x[i] = img
        i += 1  # enumerate ruined the image shape
    return augmented_x
