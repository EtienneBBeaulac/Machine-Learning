from sklearn import datasets
from node import Node
import pandas as pd
import numpy as np
import math

class NeuralLayer:
    def __init__(self, inputs, targets, num_neurons):
        self.bias_value = -1
        self.num_neurons = num_neurons
        self.num_inputs = len(inputs)
        self.inputs = inputs
        self.targets = targets
        self.nodes = [Node(self.num_inputs) for _ in range(num_neurons)]
        self.isLastLayer = False

    def calc_neurons(self, test_data):
        for neuron_id in range(self.num_neurons):
            total = self.nodes[neuron_id].weights[0] * self.bias_value
            # print(self.inputs)
            total += sum([self.nodes[neuron_id].weights[value_id + 1] * self.inputs[value_id] for value_id in range(self.num_inputs)])
            self.nodes[neuron_id].value = self.sigmoid(total)

            # print(f"\nCalculation for neuron#{neuron_id}:")
            # [print(f"{self.nodes[neuron_id].weights[value_id + 1]} * {self.inputs[value_id]} +") for value_id in range(self.num_inputs)]
            # print(f"= {total}")
            if self.nodes[neuron_id].value >= 0.5:
                self.nodes[neuron_id].fired = True
            else:
                self.nodes[neuron_id].fired = False

            print(f"Total: {self.nodes[neuron_id].value} ---> {self.nodes[neuron_id].fired}")

        self.new_inputs = [node.value for node in self.nodes]

    def show_output(self):
        print(f"{[self.nodes[neuron].fire() for neuron in range(self.num_neurons)]}")

    def sigmoid(self, x):
      return 1 / (1 + math.exp(-x))
