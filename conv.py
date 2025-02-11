import numpy as np
from scipy.signal import correlate2d
from layer import Layer
np.random.seed(6)


class Convolution(Layer):
    def __init__(self, input_shape, filter_size, num_filters):
        """
        Initializes a convolutional layer.

        Parameters:
          - input_shape: Tuple (input_depth, height, width) describing the input 
                         tensor shape (without the batch dimension).
          - filter_size: Size of the square filters.
          - num_filters: Number of filters (i.e. the output depth).
        """
        input_depth, input_height, input_width = input_shape
        self.num_filters = num_filters
        self.input_depth = input_depth

        # Filter shape: (num_filters, input_depth, filter_size, filter_size)
        self.filter_shape = (num_filters, input_depth, filter_size, filter_size)
        # Output shape per sample: (num_filters, height, width)
        self.output_shape = (num_filters, input_depth, input_height, input_width)

        # Initialize filters and biases (normalization: mean=0, std=0.4)
        self.filters = np.random.randn(*self.filter_shape) * 0.4
        # Biases are defined for each filter at each spatial location.
        self.biases = np.random.randn(*self.output_shape) * 0.4

    def forward(self, input_data):
        """
        Forward pass for the convolution layer.

        Parameters:
          - input_data: NumPy array of shape 
                        (batch_size, input_depth, height, width)

        Returns:
          - Output tensor after convolution and ReLU activation,
            of shape (batch_size, num_filters, height, width)
        """
        # Save input for use in backward pass.
        self.input_data = input_data  
        batch_size, input_depth, height, width = input_data.shape
        self.input_depth = input_depth
        # Initialize output volume.
        output = np.zeros((batch_size, self.num_filters,self.input_depth, height, width))

        # Loop over each sample in the batch.
        for n in range(batch_size):
            # For each filter.
            for f in range(self.num_filters):
                # Initialize the convolution result (for this filter and sample)
                conv_result = np.zeros((height, width))
                # Convolve over each input channel and sum the results.
                for d in range(input_depth):
                    # Pad the input (here padding=1 is assumed; e.g. for a 3x3 filter)
                    padded_input = np.pad(input_data[n, d], ((1, 1), (1, 1)), mode="constant")
                    conv_result += correlate2d(padded_input, self.filters[f, d], mode="valid")
                # Add bias and apply ReLU activation.
                output[n, f] = np.maximum(conv_result + self.biases[f], 0)
        return output

    def backward(self, dL_dout, lr):
        """
        Backward pass for the convolution layer.

        Parameters:
          - dL_dout: Gradient of the loss with respect to the output of this layer,
                     a NumPy array of shape (batch_size, num_filters, height, width).
          - lr: Learning rate for parameter updates.

        Returns:
          - dL_dinput: Gradient of the loss with respect to the input,
                       a NumPy array of shape (batch_size, input_depth, height, width).
        """
        batch_size, num_filters, height, width = dL_dout.shape

        # Allocate space for gradients w.r.t. input, filters, and biases.
        dL_dinput = np.zeros_like(self.input_data)       # (batch_size, input_depth, height, width)
        dL_dfilters = np.zeros_like(self.filters)          # (num_filters, input_depth, filter_size, filter_size)
        dL_dbiases = np.zeros_like(self.biases)            # (num_filters, height, width)

        # Loop over each sample in the batch.
        for n in range(batch_size):
            for f in range(self.num_filters):
                # Accumulate bias gradients over the batch.
                dL_dbiases[f] += dL_dout[n, f]
                for d in range(self.input_depth):
                    # Gradient with respect to filters:
                    # Correlate the input with the gradient flowing from the output.
                    dL_dfilters[f, d] += correlate2d(self.input_data[n, d], dL_dout[n, f], mode="valid")
                    
                    # Gradient with respect to the input:
                    # Pad the dL_dout for this filter and sample.
                    padded_dL = np.pad(dL_dout[n, f], ((1, 1), (1, 1)), mode="constant")
                    # Correlate the padded gradient with the corresponding filter.
                    dL_dinput[n, d] += correlate2d(padded_dL, self.filters[f, d], mode="valid")

        # Update parameters using the averaged gradients over the batch.
        self.filters -= lr * (dL_dfilters / batch_size)
        self.biases  -= lr * (dL_dbiases / batch_size)

        return dL_dinput
