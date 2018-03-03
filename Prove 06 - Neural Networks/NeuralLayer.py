from sklearn import datasets
from node import Node
import pandas as pd
import numpy as np
import math

class NeuralLayer:
    def __init__(self, num_inputs, num_neurons, is_output=False):
        self.bias_value = -1
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.nodes = [Node(self.num_inputs) for _ in range(num_neurons)]
        self.is_output = is_output

    def set(self, inputs, target):
        self.inputs = inputs
        self.target = target

    def calc_neurons_forward(self):
        for neuron_id in range(self.num_neurons):
            total = self.nodes[neuron_id].weights[0] * self.bias_value
            total += sum([self.nodes[neuron_id].weights[value_id + 1] * self.inputs[value_id] for value_id in range(self.num_inputs)])
            self.nodes[neuron_id].value = self.sigmoid(total)
            if self.nodes[neuron_id].value >= 0.5:
                self.nodes[neuron_id].fired = True
            else:
                self.nodes[neuron_id].fired = False
            # print(f"Total: {self.nodes[neuron_id].value} ---> {self.nodes[neuron_id].fired}")

        self.new_inputs = [node.value for node in self.nodes]

    def show_output(self):
        print(f"{[self.nodes[neuron].fire() for neuron in range(self.num_neurons)]}")
        print(f"{[self.nodes[neuron].value for neuron in range(self.num_neurons)]}")

    def raw_output(self):
        return [self.nodes[neuron].value for neuron in range(self.num_neurons)]

    def norm_output(self):
        return [self.nodes[neuron].fire() for neuron in range(self.num_neurons)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def error(self, right_layer=None):
        for node_id in range(self.num_neurons):
            a = self.nodes[node_id].value
            if self.is_output:
                self.nodes[node_id].error = a * (1 - a) * (a - self.target[node_id])
            else:
                self.nodes[node_id].error = a * (1 - a) * sum([node.weights[node_id + 1] for node in right_layer.nodes])
            print(f"Node #{node_id}")
            self.nodes[node_id].print_info()
            print(f"Target: {self.target[node_id]}")

    def update_weights(self, learn_rate):
        for node_id in range(self.num_neurons):
            for weight_id in range(len(self.nodes[node_id].weights)):
                # if we're at the bais weights
                if weight_id == 0:
                    self.nodes[node_id].weights[weight_id] -= learn_rate * self.nodes[node_id].error * self.bias_value
                else:
                    self.nodes[node_id].weights[weight_id] -= learn_rate * self.nodes[node_id].error * self.inputs[weight_id - 1]
            print(f"Node #{node_id}")
            print(f"\tUpdated: {self.nodes[node_id].weights}")
