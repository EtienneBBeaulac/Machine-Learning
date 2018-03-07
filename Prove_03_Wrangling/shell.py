from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from KNNClassifier import KNNClassifier
from operator import itemgetter
from wrangler import Wrangler
from sklearn.model_selection import cross_val_score, cross_val_predict



class Shell:
    def __init__(self, data, target, classifier, filename=None):
        if data is not None:
            self.data = data
        elif filename[:3] == 'csv':
            self.data = pd.read_csv(filename)
        elif filename[:3] == 'txt':
            self.data = pd.DateFrame(filename)
        else:
            print("Data Invalid")

        self.target = target

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data,
                                                                                self.target,
                                                                                test_size=0.7,
                                                                                train_size=0.3)
        self.classifier = classifier

    def shuffle_data(self):
        df_data = pd.DataFrame(self.data)
        df_target = pd.DataFrame(self.target)
        df_merged = pd.concat([df_data, df_target], axis=1)
        df_merged = shuffle(df_merged)
        self.data = df_merged.iloc[:,0:-1].as_matrix()
        self.target = df_merged.iloc[:,-1].as_matrix()

    def fit_model_to_shell(self):
        """fit the model on the data provided for the shell"""
        self.model = self.classifier.fit(self.x_train, self.y_train)


    def predict_from_classifier(self):
        """predict from the classifier and store them in the shell as y_predicted"""
        self.y_predicted = self.model.predict(self.x_test)

    def cross_val_custom(self, k=2, _range=0):
        """cross validate k times, optional range for accuracy calculation"""
        accuracies = []

        # shuffle data
        self.shuffle_data()
        # find bin size
        size = self.data.shape[0]
        bin_size = int(size / k)

        # cut the frame into bins, grab one for test, k - 1 for train
        for x in range(k):
            print(f"Cross val #{x + 1}...")
            indexes = [range((0 + (x * bin_size)), bin_size + (x * bin_size))]

            self.x_test = self.data[indexes]
            self.x_train = np.delete(self.data, indexes, axis=0)
            self.y_test = self.target[indexes]
            self.y_train = np.delete(self.target, indexes, axis=0)

            # run the normal methods
            self.fit_model_to_shell()
            self.predict_from_classifier()

            # put accuracy in list
            accuracies.append(get_accuracy(self, _range))
        print(accuracies)
        accuracies = np.array(accuracies)
        print(f"Mean accuracy is {round(np.mean(accuracies), 1)}%")

def eval_range(shell, _range, x):
    return ((shell.y_test[x] + _range <= shell.y_predicted[x]) or
            (shell.y_test[x] - _range >= shell.y_predicted[x]))


def get_accuracy(shell, _range=0):
    correct = 0
    for x in range(len(shell.y_test)):
        if _range and eval_range(shell, _range, x):
            correct += 1
        elif shell.y_test[x] == shell.y_predicted[x]:
            correct += 1
    accuracy = (correct / float(len(shell.y_test))) * 100.0
    return accuracy


"""Test Shell on normal test data and classifiers"""
classifier = KNNClassifier(k=5)

iris = datasets.load_iris()
shell = Shell(iris.data, iris.target, classifier)
shell.cross_val_custom(10)



# wrangler holds all three datasets as member variables
w = Wrangler()

# cars dataset
shell_cars = Shell(w.car_data.data.values, w.car_data.targets.values, classifier)
shell_cars.cross_val_custom(10)


# diabetes dataset
shell_d = Shell(w.diabetes_data.data.values, w.diabetes_data.targets.values, classifier)
shell_d.cross_val_custom(10)


# mpg dataset
shell_m = Shell(w.mpg_data.data.values, w.mpg_data.targets.values, classifier)
shell_m.cross_val_custom(10, 3)
