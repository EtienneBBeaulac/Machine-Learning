from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn import tree, neighbors
from wrangler import Wrangler
from shell import Shell
import pandas as pd
import numpy as np
import itertools


w = Wrangler()
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
tree = tree.DecisionTreeClassifier()
network = MLPClassifier(activation="logistic", hidden_layer_sizes=(5,))

datasets = [w.letter, w.diabetes, w.car]
classifiers = [knn, tree, network]
classifier_names = ["knn", "tree", "network"]

accuracies = []
for dataset in datasets:
    print(len(dataset.data.columns))
    for classifier, c_name in zip(classifiers, classifier_names):
        s = Shell(dataset.data, dataset.targets.values.ravel(), classifier)
        s.fit_model_to_shell()
        s.predict_from_classifier()
        accuracies.append((dataset.name, c_name, s.get_accuracy()))

for accuracy in accuracies:
    print(accuracy)

pd.DataFrame(accuracies).to_csv("test.csv")

seed = 7
results_dict = { "dataset": None, "bag": None, "ada": None, "forest": None, "num_trees": None, "max_features": None }
results_df = pd.DataFrame()
row_list = []
count = 0

for dataset in datasets:
    dataset_count = 0
    n_feat = len(dataset.data.columns)
    num_trees = np.arange(1, 200, 10)
    if dataset.name == "letter":
        max_features = np.arange(1, n_feat + 1, 2)
    else:
        max_features = np.arange(1, n_feat + 1, 1)
    c = list(itertools.product(num_trees.tolist(), max_features.tolist()))
    results_dict["dataset"] = dataset.name
    for num_tree, max_feat in c:
        results_dict["num_trees"] = num_tree
        results_dict["max_features"] = max_feat

        kfold = model_selection.KFold(n_splits=5, random_state=seed)

        """ BAGGING """
        model = BaggingClassifier(base_estimator=tree, n_estimators=num_tree, random_state=seed)
        results = model_selection.cross_val_score(model, dataset.data, dataset.targets.values.ravel().tolist(), cv=kfold)
        print(f"{dataset.name},bag,{results.mean()},{num_tree},{max_feat}")
        results_dict["bag"] = results.mean()


        """ ADABOOST """
        model = AdaBoostClassifier(n_estimators=num_tree, random_state=seed)
        results = model_selection.cross_val_score(model, dataset.data, dataset.targets.values.ravel(), cv=kfold)
        print(f"{dataset.name},ada,{results.mean()},{num_tree},{max_feat}")
        results_dict["ada"] = results.mean()


        """ RANDOM FOREST """
        model = RandomForestClassifier(n_estimators=num_tree, max_features=max_feat)
        results = model_selection.cross_val_score(model, dataset.data, dataset.targets.values.ravel(), cv=kfold)
        print(f"{dataset.name},forest,{results.mean()},{num_tree},{max_feat}")
        results_dict["forest"] = results.mean()

        print(f"iteration #{dataset_count} completed: \n{results_dict}")
        row_list.append(results_dict)
        dataset_count += 1

results_df = pd.DataFrame(row_list)
results_df.to_csv("results.csv")
