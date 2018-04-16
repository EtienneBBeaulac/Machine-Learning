from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetClassifier
from sklearn.neural_network import MLPClassifier
from wrangler import Wrangler
from sklearn import preprocessing
from sklearn.utils import shuffle
from operator import itemgetter
from sklearn import datasets
import pandas as pd
import numpy as np

class Shell:
    def __init__(self, data, target, classifier, filename=None):
        if data is not None:
            self.data = data
        elif filename[:3] == 'csv':
            self.data = pd.read_csv(filename)
        elif filename[:3] == 'txt':
            self.data = pd.DataFrame(filename)
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

            self.x_test = pd.DataFrame(self.data[indexes])
            self.x_train = pd.DataFrame(np.delete(self.data, indexes, axis=0))
            self.y_test = pd.DataFrame(self.target[indexes])
            self.y_train = pd.DataFrame(np.delete(self.target, indexes, axis=0))

            # run the normal methods
            self.fit_model_to_shell()
            self.predict_from_classifier()

            # put accuracy in list
            accuracies.append(self.get_accuracy(_range))
        print(accuracies)
        accuracies = np.array(accuracies)
        print(f"Mean accuracy is {round(np.mean(accuracies), 1)}%")

    def eval_range(self, _range, x):
        return ((self.y_test[x] + _range <= self.y_predicted[x]) or
                (self.y_test[x] - _range >= self.y_predicted[x]))

    def get_accuracy(self, _range=0):
        correct = 0
        for x in range(len(self.y_test)):
            # print(f"Target: {self.y_test.values.tolist()[x]} ---- > {self.y_predicted[x]}")
            if self.y_test.values.tolist()[x] == self.y_predicted[x]:
                correct += 1
        accuracy = (correct / float(len(self.y_test))) * 100.0
        return round(accuracy, 1)

# iris = datasets.load_iris()
# iris_data = pd.DataFrame(iris['data'],
#                      columns=iris['feature_names'])
# iris_shell = Shell(iris_data, pd.DataFrame(iris['target'],
#                                            columns=['targets']),
#                                            NeuralNetClassifier(hidden_layers_info=[5,4],
#                                            learn_rate=0.2,
#                                            epochs=200,
#                                            thres_function='sigmoid'))
# iris_shell.fit_model_to_shell()
# iris_shell.predict_from_classifier()
# print(f"Accuracy: {iris_shell.get_accuracy()}%")



w = Wrangler()

""" our implementation of iris """
iris_shell = Shell(w.iris.data,
                   w.iris.targets,
                   NeuralNetClassifier(hidden_layers_info=[5,4],
                                       learn_rate=0.2,
                                       epochs=200,
                                       thres_function='sigmoid')
                  )

iris_shell.fit_model_to_shell()
iris_shell.predict_from_classifier()
print(f"Accuracy: {iris_shell.get_accuracy()}%")

""" our implementation of diabetes """
# pima_shell = Shell(w.diabetes.data,
#                    w.diabetes.targets,
#                    NeuralNetClassifier(hidden_layers_info=[5,4],
#                    learn_rate=0.2,
#                    epochs=100,
#                    thres_function='sigmoid'))
#
# pima_shell.fit_model_to_shell()
# pima_shell.predict_from_classifier()
# print(f"Accuracy: {pima_shell.get_accuracy()}%")

""" our implementation of car """
# car_shell = Shell(w.car.data,
#                    w.car.targets,
#                    NeuralNetClassifier(hidden_layers_info=[5,4],
#                    learn_rate=0.2,
#                    epochs=100,
#                    thres_function='sigmoid'))
#
# car_shell.fit_model_to_shell()
# car_shell.predict_from_classifier()
# print(f"Accuracy: {car_shell.get_accuracy()}%")

""" existing implementation of car """
# car_clf = MLPClassifier(activation='logistic',
#                              hidden_layer_sizes=(5,))
#
# x_train, x_test, y_train, y_test = train_test_split(w.car.data,
#                                                     w.car.targets,
#                                                     test_size=0.7,
#                                                     train_size=0.3)
#
# car_clf.fit(x_train, y_train)
# print(car_clf.score(x_test, y_test))

""" existing implementation of diabetes """
# diabetes_clf = MLPClassifier(activation='logistic',
#                              hidden_layer_sizes=(5,))
#
# x_train, x_test, y_train, y_test = train_test_split(w.diabetes.data,
#                                                     w.diabetes.targets,
#                                                     test_size=0.7,
#                                                     train_size=0.3)
#
# diabetes_clf.fit(x_train, y_train)
# print(diabetes_clf.score(x_test, y_test))

""" existing implementation of iris """
# iris_clf = MLPClassifier(activation='logistic',
#                          hidden_layer_sizes=(5,))
#
# x_train, x_test, y_train, y_test = train_test_split(w.iris.data,
#                                                     w.iris.targets,
#                                                     test_size=0.7,
#                                                     train_size=0.3)
#
# iris_clf.fit(x_train, y_train)
# print(iris_clf.score(x_test, y_test))
