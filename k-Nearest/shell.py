from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from KNNClassifier import KNNClassifier
from operator import itemgetter



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


    def fit_model_to_shell(self):
        """fit the model on the data provided for the shell"""
        self.model = self.classifier.fit(self.x_train, self.y_train)


    def predict_from_classifier(self):
        """predict from the classifier and store them in the shell as y_predicted"""
        self.y_predicted = self.model.predict(self.x_test)




"""Test Shell on normal test data and classifiers"""
iris = datasets.load_iris()
classifier = KNNClassifier(k=3)

shell = Shell(iris.data, iris.target, classifier)
shell.fit_model_to_shell()
shell.predict_from_classifier()

correct = 0
for x in range(len(shell.y_test)):
	if shell.y_test[x] == shell.y_predicted[x]:
		correct += 1
accuracy = (correct / float(len(shell.y_test))) * 100.0

print(f"The model is {round(accuracy)}% correct")



# test = { 1: 12, 2: 4, 4:5, 6:10 }
# print(sorted(test.items(), key = itemgetter(1), reverse=True))






# shell.predict_from_classifier()

# """Test Hard Coded Classifier and Model"""
# hard_coded_classifier = HardCodedClassifier()
#
# new_shell = Shell(iris.data, iris.target, hard_coded_classifier)
# new_shell.fit_model_to_shell()
# new_shell.predict_from_classifier()
#
# """Accuracy check for the hardcoded model"""
# print(new_shell.y_predicted)
#
# num_correct = 0
# for predicted, target in zip(new_shell.y_predicted, new_shell.y_test):
#     if predicted == target:
#         num_correct += 1
#
# accuracy = num_correct / len(new_shell.y_test)
#
# print(len(new_shell.y_test))
# print(len(new_shell.y_predicted))
# print(f"The model is {round(accuracy * 100)}% correct")
