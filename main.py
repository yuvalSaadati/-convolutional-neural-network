# This is a sample Python script.
import numpy as np

from conv import Convolution
from data import load_data
from fully_connected import FullyConnected
from max_pool import MaxPool
from test import test_network
from train import train_network
from utils import to_categorical

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x_train, y_train = load_data("train.csv")
    y_train = to_categorical(y_train)
    layers = [Convolution(x_train[0].shape, 3, 1),
              MaxPool(2),
              Convolution((3,16,16), 3, 1),
              MaxPool(2),
              FullyConnected(input_size=192, output_size=10)]
    learning_rate = 0.00005
    epochs = 500
    batch_size = 32
    train_network(x_train, y_train,layers, learning_rate, epochs, batch_size)

    x_test, y_test = load_data("validate.csv")
    # y_test = to_categorical(y_test)
    # test_network(x_test, y_test, layers)


