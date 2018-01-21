

class HardCodedModel:
    def __init__(self):
        pass

    def predict(self, data_set):
        return [ 0 for prediction in data_set ]


class HardCodedClassifier:
    def __init__(self):
        self.model = HardCodedModel()

    def fit(self, data_train, targets_train):
        return self.model
