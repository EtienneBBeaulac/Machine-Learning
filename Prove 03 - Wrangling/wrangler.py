import pandas as pd
import numpy as np
import os
import re


class Dataset:
    """serves the purpose of abstracting out data and targets for use in kNN algorithm"""
    def __init__(self, df):
        self.targets = df["targets"]
        self.data = df.drop(["targets"], axis=1)


class Wrangler:
    def __init__(self):
        """call cleaning functions to return them as Dataset classes

        REQUIREMENT FOR CLEANING FUNCTIONS:
            - whatever column == targets, has to be set to name "targets"
        """
        # loop through files in data folder and call respective cleaning functions
        for f in os.listdir('./data/'):
            if f == "car.csv":
                self.clean_car_data("./data/" + f)
            elif f == "pima-indians-diabetes.csv":
                self.clean_diabetes_data("./data/" + f)
            elif f == "auto-mpg.data":
                self.clean_mpg_data("./data/" + f)

    def clean_car_data(self, filename):
        """CAR DATASET INFO
        targets encoding:
            - unacc (unacceptable) == 0
            - acc (acceptable) == 1
            - good == 2
            - vgood (very good) == 3
        """
        column_names = ["buying", "maint", "doors", "persons", "lug_boot",
                        "safety", "targets"]

        # temporary dataframe for cleaning and encoding
        data = pd.read_csv(filename, names=column_names)

        # check for NaN values in df
        # self.check_for_nan(data)

        # encode targets column with numerical values instead of categorical
        encoded_values = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3,
                          'low': 0, "med": 1, "high": 2, "vhigh": 3,
                          '5more': 5, 'more': 5,
                          'small': 0, 'big': 2}
        data.replace(encoded_values, inplace=True)
        data["doors"] = [int(value) for value in data.doors.values]
        data["persons"] = [int(value) for value in data.persons.values]
        # use the Dataset object to separate about data and targets
        self.car_data = Dataset(data)

    def clean_diabetes_data(self, filename):
        """DIABETES DATASET INFO
        targets encoding:
            0 == FALSE (NO DIABETES)
            1 == TRUE (POOR YOU, YOU HAVE DIABETES)
        """
        column_names = ["pregnant", "glucose", "bp", "triceps", "insulin", "bmi",
                        "pedigree", "age", "targets"]
        data = pd.read_csv(filename, names=column_names)


        cols_with_zeros = [ col for col in data.columns if col != "pregnant" or
                                                           col != "targets"]

        missing_df = data.drop(["pregnant", "targets"], axis=1)
        missing_df = missing_df.replace(0, np.NaN)
        means = missing_df.mean()
        means = np.array(means).tolist()

        column_dict = { colname: { 0: mean } for mean, colname in zip(means, cols_with_zeros) }
        data = data.replace(column_dict)

        self.diabetes_data = Dataset(data)

    def clean_mpg_data(self, filename):
        """MPG DATASET INFO
        targets encoding:
            continuous, try for a range

        """
        column_names = ["targets", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model_year", "origin", "car_name"]
        data = pd.read_csv(filename, names=column_names, na_values="?", delim_whitespace=True)

        # fill na's with the mean
        data["horsepower"] = data["horsepower"].fillna(data["horsepower"].mean())

        # for kNN, we don't need car_names
        data = data.drop(["car_name"], axis=1)
        self.mpg_data = Dataset(data)


w = Wrangler()
