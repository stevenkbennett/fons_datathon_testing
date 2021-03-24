#%% Imports + Data loading
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

feature_descriptors_headers = [*pd.read_csv('./data/train_descriptors.csv', nrows=1)]
features_descriptors = pd.read_csv(
    './data/train_descriptors.csv',
    usecols=[c for c in feature_descriptors_headers if not c in ['identifiers', 'Unnamed: 0', 'name', 'InchiKey', 'SMILES']]
)
feature_mord3d_headers = [*pd.read_csv('./data/train_mord3d.csv', nrows=1)]
features_mord3d = pd.read_csv(
    './data/train_mord3d.csv',
    usecols=[c for c in feature_mord3d_headers if not c in ['identifiers', 'Unnamed: 0', 'name', 'InchiKey', 'smiles']]
)
features = pd.concat((features_descriptors, features_mord3d), axis=1)
#features = pd.read_csv('./data/train_mol2vec.csv')
#headers = [*pd.read_csv('./data/train_rdk.csv', nrows=1)]
#features = pd.read_csv('./data/train_rdk.csv', usecols=[c for c in headers if not c in ['0']])
data = pd.read_csv('./data/train_crystals.csv')
#%% Train / test splitting
target = data['is_centrosymmetric'].to_numpy().astype(np.uint32)
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.33, random_state=42)
y_train = y_train
# %% Full model defn as pipeline
pclf = Pipeline([
    ('imputer', SimpleImputer(strategy='mean', verbose=1)),
    ('scaler', MinMaxScaler()),
    #('feature_sel', SelectKBest(f_regression, k = 100)),
    ('fitting', TruncatedSVD(2))
])
#%% Fitting
pclf.fit(X_train, y_train)
#%% Prediction
y_pred = pclf.predict(X_test)
print(f1_score(y_test, y_pred, average='macro'))
#%% plotting
#plt.plot(np.linspace(np.min(y_test), np.max(y_test)), np.linspace(np.min(y_test), np.max(y_test)), 'k-')
#plt.plot(y_test, y_pred, 'r.')
#plt.xlim([0, 50])
#plt.show()
# %%

test_descriptors_headers = [*pd.read_csv('./data/test_descriptors.csv', nrows=1)]
test_descriptors = pd.read_csv(
    './data/test_descriptors.csv',
    usecols=[c for c in test_descriptors_headers if not c in ['identifiers', 'Unnamed: 0', 'name', 'InchiKey', 'SMILES']]
)
test_mord3d_headers = [*pd.read_csv('./data/test_mord3d.csv', nrows=1)]
test_mord3d = pd.read_csv(
    './data/test_mord3d.csv',
    usecols=[c for c in test_mord3d_headers if not c in ['identifiers', 'Unnamed: 0', 'name', 'InchiKey', 'smiles']]
)
test_features = pd.concat((test_descriptors, test_mord3d), axis=1)
test_pred = np.round(pclf.predict(test_features))
#%% saving
np.savetxt('./out/task_4_predictions.csv', test_pred)
# %%
