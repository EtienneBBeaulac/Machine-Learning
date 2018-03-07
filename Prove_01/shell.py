import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from HardCodedClassifier import HardCodedModel, HardCodedClassifier


class Shell:
    def __init__(self, data, target, classifier, filename=None):
        """
        self.data:
            - either data provided by the user as parameter
            - data provided through a csv file
            - data provided through a txt file
        self.target:
            - the target data for the model
        self.classifier:
            - the chosen classifier that has the fit() method
        filename:
            - for external files
        self.predictions:
            - the predictions gathered through the model
        """
        if data is not None:
            self.data = data
        elif filename[:3] == 'csv':
            self.data = pd.read_csv(filename)
        elif filename[:3] == 'txt':
            self.data = pd.DataFrame(filename)
        else:
            print("Invalid Data")
            self.data = None

        self.target = target
        self.classifier = classifier
        self.data_train, self.data_test, self.targets_train, self.targets_test = train_test_split(self.data,
                                                                                                  self.target,
                                                                                                  test_size=0.7,
                                                                                                  train_size=0.3)

    def fit_model_to_shell(self):
        self.model = self.classifier.fit(self.data_train, self.targets_train)

    def predict_from_classifier(self):
        self.predictions = self.model.predict(self.data_test)

""" Try with GaussianNB """
iris = datasets.load_iris()
classifier = GaussianNB()
shell = Shell(iris.data, iris.target, classifier)

shell.fit_model_to_shell()
shell.predict_from_classifier()

num_correct = 0
for predicted, target in zip(shell.predictions, shell.targets_test):
    if predicted == target:
        num_correct += 1

accuracy = num_correct / len(shell.targets_test)

print(f"Accuracy of GaussianNB Model: {round(accuracy * 100)}%")

""" Test HardCodedClassifier """
hardcoded_classifier = HardCodedClassifier()
new_shell = Shell(iris.data, iris.target, hardcoded_classifier)

new_shell.fit_model_to_shell()
new_shell.predict_from_classifier()

""" Calculate accuracy """
num_correct = 0
for predicted, target in zip(new_shell.predictions, new_shell.targets_test):
    if predicted == target:
        num_correct += 1

accuracy = num_correct / len(new_shell.targets_test)

print(f"Accuracy of Hard-Coded Model: {round(accuracy * 100)}%")
