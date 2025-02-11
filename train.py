import numpy as np

from utils import cross_entropy_loss_gradient, cross_entropy_loss, augment_data

import numpy as np


def train_network(x, y, layers, lr, epochs, batch_size):
    """
    Train a network using mini-batches.
    
    Parameters:
        x         : Input data, assumed to be a NumPy array where the first dimension is the sample index.
        y         : Target labels in one-hot encoding, same first-dimension size as x.
        layers    : A list of layer objects, each with .forward() and .backward() methods.
        lr        : Learning rate.
        epochs    : Number of epochs.
        batch_size: Number of samples per batch.
    """
    n_samples = x.shape[0]

    for epoch in range(epochs):
        # Shuffle the data at the start of each epoch
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        total_loss = 0.0
        correct_predictions = 0

        # Process the dataset in mini-batches.
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_x = x_shuffled[start:end]
            batch_y = y_shuffled[start:end]

            # --------------------
            # Forward pass
            # --------------------
            # For example, if your network structure is: conv -> pool -> conv -> pool -> fully connected:
            conv1_out = layers[0].forward(batch_x)
            pool1_out = layers[1].forward(conv1_out)
            conv2_out = layers[2].forward(pool1_out)
            pool2_out = layers[3].forward(conv2_out)
            full_out  = layers[4].forward(pool2_out)  # final output: shape (batch_size, num_classes)

            # --------------------
            # Loss and Accuracy
            # --------------------
            batch_losses = cross_entropy_loss(full_out, batch_y)
            batch_loss = np.mean(batch_losses)
            total_loss += batch_loss  # weight by the number of samples in batch

            # Calculate accuracy
            predictions = np.argmax(full_out, axis=1)
            true_labels = np.argmax(batch_y, axis=1)
            correct_predictions += np.sum(predictions == true_labels)

            # --------------------
            # Backward pass
            # --------------------
            # Compute gradient of the loss with respect to the output.
            grad = cross_entropy_loss_gradient(batch_y, full_out)

            # Propagate the gradient backward through the network.
            grad = layers[4].backward(grad, lr)
            grad = layers[3].backward(grad, lr)
            grad = layers[2].backward(grad, lr)
            grad = layers[1].backward(grad, lr)
            layers[0].backward(grad, lr)

        # Compute average loss and accuracy for the epoch.
        average_loss = total_loss / n_samples
        accuracy = (correct_predictions / n_samples) * 100.0
        print(f'Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%')

