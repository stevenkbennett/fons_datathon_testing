


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import *

from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import VarianceThreshold

np.random.seed(10)

## Source data

data_folder = 'train_and_test_sets/'


train_descriptors = pd.read_csv(data_folder + "train_descriptors.csv")
train_mord3d = pd.read_csv(data_folder + "train_mord3d.csv")
train_morgan = pd.read_csv(data_folder + "train_morgan.csv")
train_rdk = pd.read_csv(data_folder + "train_rdk.csv")

train_crystals = pd.read_csv(data_folder + "train_crystals.csv")


test_descriptors = pd.read_csv(data_folder + "test_descriptors.csv")
test_mord3d = pd.read_csv(data_folder + "test_mord3d.csv")
test_morgan = pd.read_csv(data_folder + "test_morgan.csv")
test_rdk = pd.read_csv(data_folder + "test_rdk.csv")




## Data pre-processing

train_descriptors_full = train_descriptors.iloc[:, 3:-2].dropna(axis=1, how="any")

test_descriptors_full = test_descriptors[train_descriptors_full.columns]





## Lasso with CV

reg_lasso_cv = linear_model.LassoCV(cv=5, normalize=True)

reg_lasso_cv.fit(train_descriptors_full, train_crystals['calculated_density'])



predictions = reg_lasso_cv.predict(test_descriptors_full)

np.savetxt("task_1_predictions_final.csv", predictions)




