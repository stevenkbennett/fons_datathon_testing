import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import *


# Basic
import numpy as np
import pandas as pd
# Models
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# Extras
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# Scrap
import pylab
from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import eigsh
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.neighbors import NearestNeighbors
import math
from sklearn.cross_decomposition import CCA
#from sklearn.preprocessing import StandardScaler
#import matplotlib
import matplotlib.pyplot as plt


pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Models
def all_models(train_data, train_lbl, test_data): # ,test_lbl):
    ## LINEAR SVM
    # set up model
    # Scaler and regression
    pipe_linearSVM = make_pipeline(StandardScaler(), LinearSVC(tol=1e-5))
    # train
    pipe_linearSVM.fit(train_data, train_lbl)
    # predict
    pred_test_linearSVM = pipe_linearSVM.predict(test_data)
    # accuracy
#    acc_test_linearSVM = pipe_linearSVM.score(test_data, test_lbl)
#    # other evaluation measures
#    prec_linearSVM, rec_linearSVM, f1_linearSVM, sup_linearSVM = precision_recall_fscore_support(test_lbl, pred_test_linearSVM, average='weighted')
#    acc_linearSVM = accuracy_score(test_lbl, pred_test_linearSVM)

    ## LOGISTIC REGRESSION
    # set up model
    # Scaler and regression
    pipe_LogisticR = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    # train
    pipe_LogisticR.fit(train_data, train_lbl)
    # predict
    pred_test_LogisticR = pipe_LogisticR.predict(test_data)
    # accuracy
#    acc_test_LogisticR = pipe_LogisticR.score(test_data, test_lbl)
#    # other evaluation measures
#    prec_LogisticR, rec_LogisticR, f1_LogisticR, sup_LogisticR = precision_recall_fscore_support(test_lbl, pred_test_LogisticR, average='weighted')
#    acc_LogisticR = accuracy_score(test_lbl, pred_test_LogisticR)

    ## RANDOM FOREST
    # set up model
    # Scaler and regression
    pipe_RF = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=2, random_state=0))
    # train
    pipe_RF.fit(train_data, train_lbl)
    # predict
    pred_test_RF = pipe_RF.predict(test_data)
    # accuracy
#    acc_test_RF = pipe_RF.score(test_data, test_lbl)
#    # other evaluation measures
#    prec_RF, rec_RF, f1_RF, sup_RF = precision_recall_fscore_support(test_lbl, pred_test_RF, average='weighted')
#    acc_RF = accuracy_score(test_lbl, pred_test_RF)

    ## NEURAL NETWORK
    # set up model
    # Scaler and regression
    pipe_NN = make_pipeline(StandardScaler(), MLPClassifier(max_iter=100))
    # train
    pipe_NN.fit(train_data, train_lbl)
    # predict
    pred_test_NN = pipe_NN.predict(test_data)
    # accuracy
#    acc_test_NN = pipe_NN.score(test_data, test_lbl)
#    # other evaluation measures
#    prec_NN, rec_NN, f1_NN, sup_NN = precision_recall_fscore_support(test_lbl, pred_test_NN, average='weighted')
#    acc_NN = accuracy_score(test_lbl, pred_test_NN)

    ## SVM w/RBF KERNEL (SVC)
    # set up model
    # Scaler and regression
    pipe_SVC = make_pipeline(StandardScaler(), SVC(kernel = 'rbf', gamma= 'auto'))
    # train
    pipe_SVC.fit(train_data, train_lbl)
    # predict
    pred_test_SVC = pipe_SVC.predict(test_data)
    # accuracy
#    acc_test_SVC = pipe_SVC.score(test_data, test_lbl)
#    # other evaluation measures
#    prec_SVC, rec_SVC, f1_SVC, sup_SVC = precision_recall_fscore_support(test_lbl, pred_test_SVC, average='weighted')
#    acc_SVC = accuracy_score(test_lbl, pred_test_SVC)

    ## DECISION TREES
    # set up model
    # Scaler and regression
    pipe_DecisionTrees = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=0))
    # train
    pipe_DecisionTrees.fit(train_data, train_lbl)
    # predict
    pred_test_DecisionTrees = pipe_DecisionTrees.predict(test_data)
    # accuracy
#    acc_test_DecisionTrees = pipe_DecisionTrees.score(test_data, test_lbl)
#    # other evaluation measures
#    prec_DecisionTrees, rec_DecisionTrees, f1_DecisionTrees, sup_DecisionTrees = precision_recall_fscore_support(test_lbl, pred_test_DecisionTrees, average='weighted')
#    acc_DecisionTrees = accuracy_score(test_lbl, pred_test_DecisionTrees)

    ## DECISION TREES w/ ADABOOST
    # set up model
    # Scaler and regression
    pipe_DT_AdaBoost = make_pipeline(StandardScaler(), AdaBoostClassifier())
    # train
    pipe_DT_AdaBoost.fit(train_data, train_lbl)
    # predict
    pred_test_DT_AdaBoost = pipe_DT_AdaBoost.predict(test_data)
    # accuracy
#    acc_test_DT_AdaBoost = pipe_DT_AdaBoost.score(test_data, test_lbl)
#    # other evaluation measures
#    prec_DT_AdaBoost, rec_DT_AdaBoost, f1_DT_AdaBoost, sup_DT_AdaBoost = precision_recall_fscore_support(test_lbl, pred_test_DT_AdaBoost, average='weighted')
#    acc_DT_AdaBoost = accuracy_score(test_lbl, pred_test_DT_AdaBoost)

    ## LDA
    # set up model
    # Scaler and regression
    pipe_LDA = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    # train
    pipe_LDA.fit(train_data, train_lbl)
    # predict
    pred_test_LDA = pipe_LDA.predict(test_data)
    # accuracy
#    acc_test_LDA = pipe_LDA.score(test_data, test_lbl)
#    # other evaluation measures
#    prec_LDA, rec_LDA, f1_LDA, sup_LDA = precision_recall_fscore_support(test_lbl, pred_test_LDA, average='weighted')
#    acc_LDA = accuracy_score(test_lbl, pred_test_LDA)

    # K-NN
    # set up model
    # Scaler and regression
    pipe_KNN = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=10))
    # train
    pipe_KNN.fit(train_data, train_lbl)
    # predict
    pred_test_KNN = pipe_KNN.predict(test_data)
    # accuracy
#    acc_test_KNN = pipe_KNN.score(test_data, test_lbl)
#    # other evaluation measures
#    prec_KNN, rec_KNN, f1_KNN, sup_KNN = precision_recall_fscore_support(test_lbl, pred_test_KNN, average='weighted')
#    acc_KNN = accuracy_score(test_lbl, pred_test_KNN)

    ## NAIVE BAYES
    # set up model
    # Scaler and regression
    pipe_NaiveBayes = make_pipeline(StandardScaler(), GaussianNB())
    # train
    pipe_NaiveBayes.fit(train_data, train_lbl)
    # predict
    pred_test_NaiveBayes = pipe_NaiveBayes.predict(test_data)
    # accuracy
#    acc_test_NaiveBayes = pipe_NaiveBayes.score(test_data, test_lbl)
#    # other evaluation measures
#    prec_NB, rec_NB, f1_NB, sup_NB = precision_recall_fscore_support(test_lbl, pred_test_NaiveBayes, average='weighted')
#    acc_NB = accuracy_score(test_lbl, pred_test_NaiveBayes)
    # Overall scores

    return pred_test_linearSVM, pred_test_LogisticR, pred_test_RF, pred_test_NN,pred_test_SVC, pred_test_DecisionTrees, pred_test_DT_AdaBoost, pred_test_LDA, pred_test_KNN, pred_test_NaiveBayes


train_descriptors = pd.read_csv("train_and_test_sets/train_descriptors.csv")
train_mord3d = pd.read_csv("train_and_test_sets/train_mord3d.csv")
train_morgan = pd.read_csv("train_and_test_sets/train_morgan.csv")
train_rdk = pd.read_csv("train_and_test_sets/train_rdk.csv")

train_crystals = pd.read_csv("train_and_test_sets/train_crystals.csv")
train_distances = pd.read_csv("train_and_test_sets/train_distances.csv")
train_centroid_distances = pd.read_csv("train_and_test_sets/train_centroid_distances.csv")

test_descriptors = pd.read_csv("train_and_test_sets/test_descriptors.csv")
test_mord3d = pd.read_csv("train_and_test_sets/test_mord3d.csv")
test_morgan = pd.read_csv("train_and_test_sets/test_morgan.csv")
test_rdk = pd.read_csv("train_and_test_sets/test_rdk.csv")

# Data pre-processing
train_descriptors_full = train_descriptors.iloc[:, 3:-2].dropna(axis= 1, how="any")
train_descriptors_full.shape

train_mord3d_full = train_mord3d.iloc[:, 3:-2].dropna(axis= 1, how="any")
print(train_mord3d_full.shape)
train_morgan_full = train_morgan.iloc[:, 3:-2].dropna(axis= 1, how="any")
print(train_morgan_full.shape)
train_rdk_full = train_morgan.iloc[:, 3:-2].dropna(axis= 1, how="any")
print(train_rdk_full.shape)

test_descriptors_full = test_descriptors[train_descriptors_full.columns]
print(test_descriptors_full.shape)
test_mord3d_full = test_mord3d[train_mord3d_full.columns]
print(test_mord3d_full.shape)
test_morgan_full = test_morgan[train_morgan_full.columns]
print(test_morgan_full.shape)
test_rdk_full = test_rdk[train_rdk_full.columns]
print(test_rdk_full.shape)

# Response
is_crystal = train_crystals.loc[:,'is_centrosymmetric']

# PCA
# Select data
train_PCA = decomposition.PCA(n_components=.95)
scaler_for_PCA = preprocessing.StandardScaler()
train_data_PCA = train_PCA.fit_transform(scaler_for_PCA.fit_transform(train_descriptors_full))
test_data_PCA = train_PCA.transform(scaler_for_PCA.transform(test_descriptors_full))
print(train_data_PCA.shape, test_data_PCA.shape)

# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
train_data = train_data_PCA
train_lbl = is_crystal
test_data = test_data_PCA


pred_linearSVM, pred_LogisticR, pred_RF, pred_NN, pred_SVC, pred_DT, pred_AdaBoost, pred_LDA, pred_KNN, pred_NB = all_models(train_data = train_data_PCA,
                         train_lbl = train_lbl,
                         test_data = test_data_PCA)

# Save for a selected model
# example pred_linearSVM
pd.DataFrame(pred_linearSVM).to_csv("task_2_predictions.csv")
