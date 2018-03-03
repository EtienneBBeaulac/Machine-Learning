import numpy as np

class Node:
    def __init__(self, num_inputs):
        self.weights = np.random.uniform(low=-1, high=1, size=num_inputs + 1)
        self.fired = False
        self.value = None
        self.error = None

    def fire(self):
        if self.fired is False:
            return 0
        else:
            return 1

    def print_info(self):
        print(f"\tWeights: {self.weights}")
        print(f"\tValue: {self.value}")
        print(f"\tError: {self.error}")
