from sklearn import datasets
from node import Node
import pandas as pd
import numpy as np
import math

class NeuralLayer:
    def __init__(self, num_inputs, num_neurons, bias_value):
        self.bias_value = bias_value
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.nodes = [Node(self.num_inputs) for _ in range(num_neurons)]

    def set(self, inputs, target, is_output=False):
        self.inputs = inputs
        self.target = target
        self.is_output = is_output

    def calc_neurons_forward(self):
        # Go through each neuron in layer
        for neuron_id in range(self.num_neurons):
            # Get the h value
            total = self.nodes[neuron_id].weights[0] * self.bias_value
            total += sum([self.nodes[neuron_id].weights[value_id + 1] * self.inputs[value_id] for value_id in range(self.num_inputs)])
            # Get the activation value
            self.nodes[neuron_id].value = self.sigmoid(total)
            # See if they fire or not
            if self.nodes[neuron_id].value >= 0.5:
                self.nodes[neuron_id].fired = True
            else:
                self.nodes[neuron_id].fired = False
        # The inputs of the next layer
        self.new_inputs = [node.value for node in self.nodes]

    def describe(self):
        # For debugging
        print(f"Number of inputs: {self.num_inputs}")
        print(f"Number of neurons: {self.num_neurons}")
        print(f"Inputs: {self.inputs}")
        print(f"Target: {self.target}")
        print(f"Output? {self.is_output}")
        for node_id in range(len(self.nodes)):
            print(f"Node #{node_id}")
            self.nodes[node_id].print_info()

    def raw_output(self):
        return [self.nodes[neuron].value for neuron in range(self.num_neurons)]

    def norm_output(self):
        return [self.nodes[neuron].fire() for neuron in range(self.num_neurons)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def error(self, right_layer=None):
        # Calculate the error of each node
        for node_id in range(self.num_neurons):
            a = self.nodes[node_id].value
            if self.is_output:
                # Output error formula a(1 - a)(a - t)
                self.nodes[node_id].error = a * (1 - a) * (a - self.target[node_id])
            else:
                # Hidden layer formula a(1 - a)∑wζ
                self.nodes[node_id].error = a * (1 - a) * sum([node.weights[node_id + 1] * node.error for node in right_layer.nodes])

    def update_weights(self, learn_rate):
        for node_id in range(self.num_neurons):
            for weight_id in range(len(self.nodes[node_id].weights)):
                # if we're at the bias weights
                if weight_id == 0:
                    self.nodes[node_id].weights[weight_id] -= learn_rate * self.nodes[node_id].error * self.bias_value
                else:
                    self.nodes[node_id].weights[weight_id] -= learn_rate * self.nodes[node_id].error * self.inputs[weight_id - 1]
