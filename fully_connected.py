import numpy as np
from layer import Layer
from utils import softmax, softmax_derivative

np.random.seed(6)


class FullyConnected(Layer):
    def __init__(self, input_size, output_size):
        """
        Parameters:
          - input_size: The flattened size of each input sample.
          - output_size: The number of neurons in this layer (number of output classes).
        """
        self.input_size = input_size   # Expected size after flattening each sample.
        self.output_size = output_size
        # Initialize weights of shape (output_size, input_size)
        self.weights = np.random.randn(output_size, input_size) * 0.4
        # Initialize biases as column vectors of shape (output_size, 1)
        self.biases = np.random.rand(output_size, 1) * 0.4

        # These will be set during forward pass.
        self.input_data = None  # The original (batched) input data.
        self.z = None         # The pre-activation (weighted input) for each sample.
        self.output = None    # The output after softmax.

    def forward(self, input_data):
        """
        Forward pass for batched inputs.

        Parameters:
          - input_data: NumPy array with shape (batch_size, ...). Each sample will be flattened.

        Returns:
          - output: NumPy array of shape (batch_size, output_size), representing the softmax
                    probabilities for each sample.
        """
        self.input_data = input_data
        batch_size = input_data.shape[0]
        # Flatten each sample: resulting shape (batch_size, input_size)
        flattened_input = input_data.reshape(batch_size, -1)
        # Compute pre-activation z.
        # Using weights of shape (output_size, input_size), we compute:
        #    flattened_input (batch_size, input_size) dot weights.T (input_size, output_size)
        # which results in (batch_size, output_size).
        # Biases are stored as (output_size, 1), so we add biases.T (shape: (1, output_size))
        # and broadcasting takes care of adding them to every sample.
        self.z = np.dot(flattened_input, self.weights.T) + self.biases.T
        # Apply softmax activation to each sample.
        self.output = softmax(self.z)
        return self.output

    def backward(self, dL_dout, lr):
        """
        Backward pass for the fully connected layer.

        Parameters:
          - dL_dout: Gradient of the loss with respect to the output of this layer,
                     of shape (batch_size, output_size).
          - lr: Learning rate for parameter updates.

        Returns:
          - dL_dinput: Gradient of the loss with respect to the input data,
                       with the same shape as self.input_data.
        """
        batch_size = self.input_data.shape[0]
        # Initialize gradient w.r.t. the pre-activation z.
        # dL_dz will have shape (batch_size, output_size)
        dL_dz = np.zeros_like(self.output)
        # For each sample, compute the Jacobian of the softmax and use it to backpropagate.
        for i in range(batch_size):
            jacobian = softmax_derivative(self.output[i])  # Expected shape: (output_size, output_size)
            dL_dz[i] = np.dot(jacobian, dL_dout[i])
        
        # Flatten the input data for use in gradient computation.
        flattened_input = self.input_data.reshape(batch_size, -1)  # Shape: (batch_size, input_size)

        # Compute the gradient with respect to the weights.
        # Using: dL_dw = dL_dz.T (shape: (output_size, batch_size)) dot flattened_input (batch_size, input_size)
        # This yields dL_dw of shape (output_size, input_size).
        dL_dw = np.dot(dL_dz.T, flattened_input)
        
        # Compute the gradient with respect to the biases.
        # Sum over the batch dimension. The result is (1, output_size), so we transpose to (output_size, 1)
        dL_db = np.sum(dL_dz, axis=0, keepdims=True).T
        
        # Compute the gradient with respect to the input.
        # dL_dinput_flat: (batch_size, input_size) computed by: dL_dz (batch_size, output_size) dot weights (output_size, input_size)
        dL_dinput_flat = np.dot(dL_dz, self.weights)
        # Reshape to the original input shape.
        dL_dinput = dL_dinput_flat.reshape(self.input_data.shape)

        # Update weights and biases using the learning rate.
        self.weights -= lr * dL_dw
        self.biases  -= lr * dL_db

        return dL_dinput
