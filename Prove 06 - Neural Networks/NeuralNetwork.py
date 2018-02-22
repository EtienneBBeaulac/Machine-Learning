from NeuralLayer import NeuralLayer
from sklearn import datasets
from node import Node
import pandas as pd
import numpy as np

class NeuralNetModel:
    def __init__(self, inputs, targets, num_neurons, num_layers):
        self.bias_value = -1
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.num_inputs = len(inputs.columns)
        self.inputs = inputs
        self.targets = targets
        # self.layers = [NeuralLayer(self.num_inputs) for _ in range(num_layers)]

    def predict(self, test_data):
        for index, row in self.inputs.iterrows():
            self.layers = []
            for x in range(self.num_layers):
                print(f"-------------Layer #{x} for row #{index}---------------")
                if x == 0:
                    self.layers.append(NeuralLayer(row, self.targets, self.num_neurons))
                    self.layers[x].calc_neurons(test_data)
                else:
                    self.layers.append(NeuralLayer(self.layers[x - 1].new_inputs, self.targets, self.num_neurons))
                    self.layers[x].calc_neurons(test_data)
                if x == self.num_layers - 1:
                    self.layers[x].show_output()

class NeuralNetClassifier:
    def __init__(self, num_neurons=None, num_layers=1):
        self.num_neurons = num_neurons
        self.num_layers = num_layers

    def fit(self, data_train, targets_train):
        if self.num_neurons is None:
            return NeuralNetModel(data_train, targets_train, targets_train.nunique()[0], self.num_layers)
        else:
            return NeuralNetModel(data_train, targets_train, self.num_neurons, self.num_layers)
