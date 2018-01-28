import numpy as np
from sklearn import datasets
from operator import itemgetter


class KNNModel:
    def __init__(self, data_train, targets_train, k):
        self.data_train = data_train
        self.targets_train = targets_train
        self.k = k

    def compute_distances(self, data, test_row):
        return [np.sum(self.get_squared_diff(row, test_row)) for row in data]

    def get_squared_diff(self, x1, x2):
        return [ (x1_point - x2_point)**2 for x1_point, x2_point in zip(x1, x2) ]

    def get_sorted_distances(self, data_train, data_test_row, targets_train):
        distances = self.compute_distances(data_train, data_test_row)
        indexed_dist = [ (i, distance) for i, distance in zip(targets_train, distances) ]
        return sorted(indexed_dist, key=itemgetter(1))

    def vote(self, distances):
        results = {}
        for n in range(self.k):
            result = distances[n][0]
            if result in results:
                results[result] += 1
            else:
                results[result] = 1

            return sorted(results.items(), key=itemgetter(1), reverse=True)[0][0]

    def predict(self, data_test):
        predictions = []
        for x in range(len(data_test)):
            distances = self.get_sorted_distances(self.data_train, data_test[x], self.targets_train)
            vote = self.vote(distances)
            predictions.append(vote)
        return predictions


class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, data_train, targets_train):
        return KNNModel(data_train, targets_train, self.k)
