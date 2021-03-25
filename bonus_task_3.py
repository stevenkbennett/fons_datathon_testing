#! /usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn import *

## Training
train_descriptors = pd.read_csv("../train_and_test_sets/train_descriptors.csv")
train_mord3d = pd.read_csv("../train_and_test_sets/train_mord3d.csv")
train_morgan = pd.read_csv("../train_and_test_sets/train_morgan.csv")
train_rdk = pd.read_csv("../train_and_test_sets/train_rdk.csv")

## Train responses
train_crystals = pd.read_csv("../train_and_test_sets/train_crystals.csv")
train_distances = pd.read_csv("../train_and_test_sets/train_distances.csv")
train_centroid_distances = pd.read_csv("../train_and_test_sets/train_centroid_distances.csv")

train_descriptors_full = train_descriptors.iloc[:, 3:-2].dropna(axis=1, how="any")
train_mord3d_full = train_mord3d.dropna(axis=1, how="any").drop(['identifiers','Unnamed: 0', 'InchiKey', 
                                                                 'smiles', 'name'], axis=1)
train_morgan_full = train_morgan.drop('0',axis=1)
train_rdk = train_rdk.drop('0',axis=1)

## Testing
test_descriptors = pd.read_csv("../train_and_test_sets/test_descriptors.csv")
test_mord3d = pd.read_csv("../train_and_test_sets/test_mord3d.csv")
test_morgan = pd.read_csv("../train_and_test_sets/test_morgan.csv")
test_rdk = pd.read_csv("../train_and_test_sets/test_rdk.csv")

test_descriptors_full = test_descriptors[train_descriptors_full.columns]
test_mord3d_full = test_mord3d[train_mord3d_full.columns]
test_morgan_full = test_morgan.drop('0',axis=1)
test_rdk = test_rdk.drop('0',axis=1)

target = 'spacegroup_symbol'

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0, n_estimators=10)
clf.fit(train_data, train_crystals[target])

predictions = clf.predict(train_data)

f1_score(predictions, train_crystals[target], average='macro')

predictions = clf.predict(test_data)
pd.DataFrame(predictions).to_csv("task_spacegroup_symbol_random_forestt.csv", header=False, index=False)
