from NeuralLayer import NeuralLayer
from sklearn import datasets
from node import Node
import pandas as pd
import numpy as np

class NeuralNetModel:
    def __init__(self, layers, new_targets):
        self.layers = layers
        self.new_targets = new_targets
        self.classifications = []
        pass

    def predict(self, test_data):
        self.inputs = test_data
        self.forward()
        return self.classifications

    def forward(self):
        # Go through each row in the dataset
        for index, row in self.inputs.iterrows():
            # Go through each layer
            for x in range(len(self.layers)):
                # print(f"-------------Layer #{x} for row #{index} PREDICT---------------")
                # This is for input layer
                if x == 0:
                    self.layers[x].inputs = row
                    self.layers[x].calc_neurons_forward()
                else:
                    self.layers[x].inputs = self.layers[x - 1].new_inputs
                    self.layers[x].calc_neurons_forward()
                # Show output
                if self.layers[x].is_output:
                    # self.layers[x].show_output()
                    # print(self.classify())
                    self.classifications.append([self.classify()])

    def classify(self):
        if tuple(self.layers[-1].norm_output()) in self.new_targets:
            return self.new_targets[tuple(self.layers[-1].norm_output())]
        else:
            return -1

class NeuralNetClassifier:
    def __init__(self, hidden_layers_info=None, bias_value=-1, learn_rate=0.1, epochs=50):
        self.hidden_layers_info = hidden_layers_info
        self.bias_value = bias_value
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.classifications = []

    def fit(self, data_train, targets_train):
        # Find some good values for number of layers and number of nodes at each layer
        self.num_outputs = targets_train.nunique()[0]
        self.num_inputs = len(data_train.columns)
        if self.hidden_layers_info is None:
            self.hidden_layers_info = [self.num_inputs, self.num_outputs]
        # Add an output layer the size of the number of targets
        self.hidden_layers_info.append(targets_train.nunique()[0])
        self.inputs = data_train
        self.targets = pd.DataFrame(targets_train)
        self.new_targets_k = {self.generate_target(target):target
                              for target in self.targets['targets'].unique().tolist()}
        self.new_targets_v = {target:self.generate_target(target)
                              for target in self.targets['targets'].unique().tolist()}
        self.layers = [NeuralLayer(num_inputs=self.num_inputs, num_neurons=self.hidden_layers_info[layer_id]) if layer_id == 0
                       else NeuralLayer(num_inputs=self.hidden_layers_info[layer_id - 1], num_neurons=self.hidden_layers_info[layer_id], is_output=True)
                       for layer_id in range(len(self.hidden_layers_info))]
        # Go through each row in the dataset
        for index, row in self.inputs.iterrows():
            self.forward(index, row)
            self.backward(index, row)
        print(f"{self.layers[-1].nodes[0].weights}")
        return NeuralNetModel(self.layers, self.new_targets_k)

    def forward(self, index, row):
        # self.layers = []
        # Go through each layer
        for x in range(len(self.hidden_layers_info)):
            print(f"-------------Layer #{x} for row #{index}---------------")
            # Check if we're at the output layer
            is_output = x == len(self.hidden_layers_info) - 1
            # This is for input layer
            if x == 0:
                # self.layers.append(NeuralLayer(
                #     row,
                #     self.new_targets_v[self.targets.loc[index].values[0]],
                #     self.hidden_layers_info[x],
                #     is_output)
                # )
                self.layers[x].set(inputs=row,
                                   target=self.new_targets_v[self.targets.loc[index].values[0]])
                self.layers[x].calc_neurons_forward()
            # Other layers
            else:
                # self.layers.append(NeuralLayer(
                #     self.layers[x - 1].new_inputs,
                #     self.new_targets_v[self.targets.loc[index].values[0]],
                #     self.hidden_layers_info[x], is_output)
                # )
                self.layers[x].set(inputs=self.layers[x - 1].new_inputs,
                                   target=self.new_targets_v[self.targets.loc[index].values[0]])
                self.layers[x].calc_neurons_forward()
            # Show output
            if is_output:
                self.layers[x].show_output()
                self.classifications.append(self.classify())

    def backward(self, index, row):
        for x in reversed(range(len(self.layers))):
            print(f"Layer #{x}")
            print( "----------------------------")
            if x == len(self.layers) - 1:
                print("Output error")
                self.layers[x].error()
            else:
                print("Hidden error")
                self.layers[x].error(self.layers[x + 1])
            self.layers[x].update_weights(self.learn_rate)

    def classify(self):
        if tuple(self.layers[-1].norm_output()) in self.new_targets_k:
            return self.new_targets_k[tuple(self.layers[-1].norm_output())]
        else:
            return -1

    def generate_target(self, target):
        temp = [0] * self.num_outputs
        temp[target] = 1
        return tuple(temp)
