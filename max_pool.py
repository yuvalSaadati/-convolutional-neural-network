import numpy as np
from layer import Layer

class MaxPool(Layer):
    def __init__(self, pool_size):
        self.pool_size = pool_size
        # These attributes will be defined during the forward pass.
        self.input_data = None
        self.num_channels = None
        self.input_height = None
        self.input_width = None
        self.output_height = None
        self.output_width = None
        self.num_filters = None

    def forward(self, input_data):
        """
        Performs the forward pass of the MaxPool layer on batched input.

        Parameters:
            input_data : np.array
                Shape (batch_size, num_channels, input_height, input_width)

        Returns:
            output : np.array
                Shape (batch_size, num_channels, output_height, output_width)
        """
        # Save input for the backward pass.
        input_data = np.squeeze(input_data, axis=1) # TODO: handle more then one filter
        self.input_data = input_data
        batch_size,  self.num_channels, self.input_height, self.input_width = input_data.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        # Initialize the output volume.
        output = np.zeros((batch_size, self.num_channels, self.output_height, self.output_width))

        # Loop over each sample in the batch.
        for n in range(batch_size):
            # Loop over each channel.
            for c in range(self.num_channels):
                # Loop over the output height.
                for i in range(self.output_height):
                    # Loop over the output width.
                    for j in range(self.output_width):
                        # Determine the boundaries of the current pooling window.
                        start_i = i * self.pool_size
                        start_j = j * self.pool_size
                        end_i = start_i + self.pool_size
                        end_j = start_j + self.pool_size

                        # Extract the current patch.
                        patch = input_data[n, c, start_i:end_i, start_j:end_j]
                        # Compute the maximum value in the patch.
                        output[n, c, i, j] = np.max(patch)

        return output

    def backward(self, dL_dout, lr):
        """
        Performs the backward pass of the MaxPool layer.

        Parameters:
            dL_dout : np.array
                Gradient of the loss with respect to the output of the max pooling layer.
                Shape (batch_size, num_channels, output_height, output_width)
            lr : float
                Learning rate (not used in pooling but included for API consistency).

        Returns:
            dL_dinput : np.array
                Gradient of the loss with respect to the input of the max pooling layer.
                Shape (batch_size, num_channels, input_height, input_width)
        """
        batch_size = self.input_data.shape[0]
        # Initialize the gradient with respect to the input.
        dL_dinput = np.zeros_like(self.input_data)

        # Loop over each sample in the batch.
        for n in range(batch_size):
            # Loop over each channel.
            for c in range(self.num_channels):
                # Loop over the pooled output's height.
                for i in range(self.output_height):
                    # Loop over the pooled output's width.
                    for j in range(self.output_width):
                        # Determine the boundaries of the current pooling window.
                        start_i = i * self.pool_size
                        start_j = j * self.pool_size
                        end_i = start_i + self.pool_size
                        end_j = start_j + self.pool_size

                        # Extract the current patch from the input.
                        patch = self.input_data[n, c, start_i:end_i, start_j:end_j]
                        # Create a mask that identifies the maximum element(s) in the patch.
                        mask = patch == np.max(patch)
                        # Distribute the gradient from the output to the positions of the maximum value(s).
                        dL_dinput[n, c, start_i:end_i, start_j:end_j] = dL_dout[n, c, i, j] * mask

        return dL_dinput
