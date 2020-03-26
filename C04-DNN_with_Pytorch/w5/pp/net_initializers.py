"""
A set of utility classes/functions for jupyter (pytorch) notebooks
"""

from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNet(nn.Module, ABC):
    """
    The parent class
    """
    ## Constructor
    def __init__(self, layers):
        super().__init__()
        self.hidden = nn.ModuleList()
        self._init_layers(layers)

    ## Prediction
    def forward(self, x):
        num_hidden = len(self.hidden)
        for (lnum, linear_transform) in zip(range(num_hidden), self.hidden):
            if lnum < num_hidden - 1: # not the last layer
                x = F.relu(linear_transform(x))
            else:         # last layer
                x = linear_transform(x)
        return x

    def _init_layers(self, layers):
        pass

class NetHe(BaseNet):
    """
    The class for neural network model with He Initialization
    """
    ## Constructor
    def __init__(self, layers):
        super().__init__(layers)

    ## Prediction => superclass

    ## private
    def _init_layers(self, layers):
        for input_size, output_size in zip(layers, layers[1:]):
            linear = nn.Linear(input_size, output_size)
            # using He init:
            torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            self.hidden.append(linear)

class NetUniform(BaseNet):
    """
    The class for neural network model with Uniform Initialization
    """
    ## Constructor
    def __init__(self, layers):
        super().__init__(layers)

    ## Prediction => superclass

    ## private
    def _init_layers(self, layers):
        for input_size, output_size in zip(layers, layers[1:]):
            linear = nn.Linear(input_size, output_size)
            linear.weight.data.uniform_(0, 1)
            self.hidden.append(linear)

class Net(BaseNet):
    """
    The class for neural network model with PyTorch Default Initialization
    """
    ## Constructor
    def __init__(self, layers):
        super().__init__(layers)

    ## Prediction => superclass

    ## private
    def _init_layers(self, layers):
        for input_size, output_size in zip(layers, layers[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden.append(linear)

class NetXavier(BaseNet):
    """
    The class for neural network model with Xavier Initialization
    """
    ## Constructor
    def __init__(self, layers):
        super().__init__(layers)

    ## Prediction => superclass

    ## private
    def _init_layers(self, layers):
        for input_size, output_size in zip(layers, layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.xavier_uniform_(linear.weight)
            self.hidden.append(linear)
