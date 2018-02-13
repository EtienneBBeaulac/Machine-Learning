import numpy as np
import pandas as pd
from sklearn import preprocessing
import pprint


class ID3Model:
    def __init__(self, data_train, targets_train):
        if not isinstance(data_train, pd.DataFrame):
            data_train = pd.DataFrame(data_train)
        if not isinstance(targets_train, pd.DataFrame):
            targets_train = pd.DataFrame(targets_train, columns=['targets'])

        self.dataset = pd.concat([data_train, targets_train], axis=1)

        feature_names = self.dataset.drop(["targets"], axis=1).columns.values

        self.unique_vals_df = pd.DataFrame()
        for col in self.dataset.columns:
            self.unique_vals_df[col] = pd.Series(self.dataset[col].unique())

        self.tree = self.create_tree(self.dataset, feature_names)

    def show_tree(self):
        """use prettyprinter to show the tree in readable format"""
        pp = pprint.PrettyPrinter()
        pp.pprint(self.tree)

    def create_tree(self, df, features):
        """recursively create dictionary tree
        - uses info gain to determine best features to split on
        """
        data = df.copy()
        values = data[features]
        default = data["targets"].value_counts().index.tolist()[0]

        if len(features) == 0:
            return default
        elif len(set(data["targets"])) == 1:
            return data["targets"].iloc[0]
        else:
            gain_dict = {}
            for feature in features:
                gain_dict[feature] = self.calc_info_gain(feature, data)
            best_feature = max(gain_dict, key=gain_dict.get)

            unique_values = np.unique(data[best_feature])

            missing_values = pd.DataFrame()
            missing_values = pd.concat([pd.DataFrame(data[best_feature].unique()), pd.DataFrame(self.unique_vals_df[best_feature])], ignore_index=True, axis=1)
            missing_values = pd.DataFrame(pd.concat([missing_values[0], missing_values[1]], ignore_index=True)).dropna()
            missing_values = missing_values.drop_duplicates(keep=False)

            tree = { best_feature: {} }
            if missing_values[0].count() > 0:
                for value in missing_values[0]:
                    tree[best_feature][value] = default

            for value in unique_values:
                new_data = data[data[best_feature] == value].drop([best_feature], axis=1)
                subtree = self.create_tree(new_data, new_data.drop(["targets"], axis=1).columns)
                tree[best_feature][value] = subtree

            return tree


    def calc_info_gain(self, feature, df):
        """calculate info gain of individual features"""
        # create one df for every column in self.dataset, grouped by column and targets
        grouped_data = df.groupby([feature, "targets"])\
                         .size().reset_index().rename(columns={0:"count"})

        cross_data = pd.crosstab(index=df["targets"], columns=df[feature])

        feature_count = grouped_data["count"].sum()
        entropy_list = []

        # Calculate the entropy of every value in feature
        unique_values = np.unique(grouped_data[feature])
        # get a list of unique target values
        unique_targets = np.unique(df["targets"])
        for value in unique_values:
            # Subset values
            data = grouped_data[grouped_data[feature] == value]
            # Calc total target count for value
            value_count = data["count"].sum()

            entropy_values = []
            for target in unique_targets:
                prob = data[data["targets"] == target]["count"].sum() / value_count
                if not pd.isnull(prob):
                    entropy_values.append(self.calc_entropy(prob))

            entropy_list.append(np.sum(entropy_values) * (value_count / feature_count))

        row_count = df.shape[0]
        df_value_counts = df["targets"].value_counts()

        feature_entropy = np.sum(entropy_list)
        total_entropy = np.sum([self.calc_entropy(df_value_counts[target] / row_count)
                                                  for target in unique_targets])

        return total_entropy - feature_entropy

    def calc_entropy(self, p):
        """Algorithm calculating entropy from Machine Learning, an Algorithmic Perspective
        credit for function: Stephen Marsland
        """
        if p != 0:
            return -p * np.log2(p)
        else:
            return 0


    def get_leaf(self, node, data=pd.DataFrame):
        """recursively go through tree to find the leaf node of given data"""
        if not isinstance(node, dict):
            return node
        for k,v in node.items():
            my_value = data[k].values[0]
            new_node = v[my_value]
            # next node taken from the next feature dictionary
            return self.get_leaf(new_node, data)

    def predict(self, test_data):
        """return a list of leaf nodes (predicted values) for any given set of test data"""
        return [ self.get_leaf(self.tree, test_data.loc[[index]])
                 for index, row in test_data.iterrows() ]


class ID3DecisionTree:
    def __init__(self):
        pass

    def fit(self, data_train, targets_train):
        return ID3Model(data_train, targets_train)


############# TEST ###############

# data = {"Weather": ["Hot", "Cold", "Nice", "Cold", "Hot", "Hot", "Cold", "Nice", "Nice", "Hot", "Nice"],
#          "Test": ["Pass", "Pass", "Fail", "Pass", "Fail", "Fail", "Pass", "Fail", "Pass", "Pass", "Pass" ],
#          "Chocolate": ["Dark", "Dark", "Dark", "Dark", "Dark", "Dark", "Dark", "Dark", "Dark", "Dark", "Dark"],
#          "Cholo": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] }
# targets = [1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0]
