from sklearn import datasets
import pandas as pd
import numpy as np

iris = datasets.load_iris()
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                    columns= iris['feature_names'] + ['target'])

new_data = pd.DataFrame()
for feature in data.columns:
    if feature != "target":
        new_data[feature] = pd.cut(data[feature], bins = 3).astype(str).apply(str)
        print(new_data[feature].apply(type))

print(new_data.head)
