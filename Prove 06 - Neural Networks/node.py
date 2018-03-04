import numpy as np

class Node:
    def __init__(self, num_inputs, thres_function):
        self.weights = np.random.uniform(low=-1, high=1, size=num_inputs + 1)
        self.fired = False
        self.value = None
        self.error = None
        self.thres_function = thres_function

    def fire(self):
        if self.thres_function == 'sigmoid':
            if self.fired is False:
                return 0
            else:
                return 1
        elif self.thres_function == 'softsign' or self.thres_function == 'tanh':
            if self.fired is False:
                return -1
            else:
                return 1

    def print_info(self):
        print(f"\tWeights: {self.weights}")
        print(f"\tValue: {self.value}")
        print(f"\tError: {self.error}")
