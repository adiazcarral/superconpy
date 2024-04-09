#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:47:15 2023

@author: mroitegui
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tsfel
from sklearn.decomposition import PCA, KernelPCA
import seaborn as sb
import re
from scipy import ndimage, misc
from scipy.stats import entropy
from scipy import signal
from scipy import constants
from scipy.fftpack import fft, dct
from sklearn import manifold
from sklearn.manifold import TSNE, MDS
import xgboost
import sklearn.metrics
import math
from matplotlib import pyplot as plt

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# Import custom modules
from vecs import compute_vecs
import featurefunctions as F

# Read CSV files
zhang_train = pd.read_csv("data/12340_all_pred.csv")
cuprates = pd.read_csv('martin/cupratos.csv')
ironbase = pd.read_csv('martin/ironbase.csv')
lowTcuprate = pd.read_csv('martin/lowT.csv')
highTcuprate = pd.read_csv('martin/highT.csv')

# Define the CSV file path for 3DSC
csv_path = "/data/angel/Superconductivity/3DSC/superconductors_3D/data/final/MP/3DSC_MP.csv"
# csv_path = "/Users/Angel/data/Superconductivity/3DSC/superconductors_3D/data/final/MP/3DSC_MP.csv"

# Read the CSV file using pandas, skipping the first row for data and using the second row for column names
df_3DSC = pd.read_csv(csv_path, skiprows=[0])
electrones = pd.read_csv("data/periodic_table_of_elementswithelectronstotal.csv")
superconductors_list = df_3DSC['formula_sc'].tolist()

# Convert categorical variable to numerical using LabelEncoder
label_encoder = LabelEncoder()

# Formula for 3DSC dataset
formula = pd.read_csv(csv_path, skiprows=[0])

# FEATURE EXTRACTION
EN = F.electrodifference(formula, electrones, superconductors_list)
Mval = F.mval(formula, electrones, superconductors_list)
Mtc = F.mtc(formula, electrones, superconductors_list)
Maradius = F.mrad(formula, electrones, superconductors_list)
Mfie = F.mfie(formula, electrones, superconductors_list)
Mec = F.mec(formula, electrones, superconductors_list)
Masa = F.masa(formula, electrones, superconductors_list)
densidads, densidadp, densidadd, densidadf = F.eletronicdensity(formula, electrones, superconductors_list)
Mend = F.mend(formula, electrones, superconductors_list)
vecs = F.compute_vecs(formula, electrones, superconductors_list)
electronegativity_list = F.EN_old(formula, electrones, superconductors_list)
smix = F.mixing_entropy(superconductors_list)
Delta = F.delta(electrones, superconductors_list)

# Extracting lattice vectors, volume, and symmetry group
lattice_vector_a = df_3DSC['lata_2'].values.reshape(-1, 1)
lattice_vector_b = df_3DSC['latb_2'].values.reshape(-1, 1)
lattice_vector_c = df_3DSC['latc_2'].values.reshape(-1, 1)
density = df_3DSC['density_2'].values.reshape(-1, 1)
crystal_system = label_encoder.fit_transform(df_3DSC['crystal_system_2'].values).reshape(-1, 1)
point_group = label_encoder.fit_transform(df_3DSC['point_group_2'].values).reshape(-1, 1)

# Convert lattice vectors to a single array
lattice_vectors = np.column_stack((lattice_vector_a, lattice_vector_b, lattice_vector_c))

# Combine the extracted information into a numpy array
superconductor_data = np.column_stack((lattice_vectors, density, crystal_system))

# Drop outliers based on smix
outliers_smix = np.argwhere(np.isnan(smix))
formula.drop(outliers_smix[:, 0], axis=0, inplace=True)

# Define features
features = [
    '1s', '2s', '2p', '3s', '3p', '3d', '4s', '4p', '4d', '4f', '5s', '5p', '5d', '5f', '6s', '6p', '6d', '6f', '7s',
    '7p', 'Mend', 'Mass', 'EN', 'Maradius', 'Mval', 'Mtc', 'Mfie', 'Mec', 'Smix', 'delta'
]

# Create feature matrix
X = np.concatenate((
    vecs[:, [0, 4, 5, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]],
    Mend, Masa, EN, Maradius, Mval, Mtc, Mfie, Mec, smix, Delta,
    lattice_vector_a, lattice_vector_b, lattice_vector_c, density, crystal_system, point_group
), axis=1)
print(np.shape(X))
# Labels
yy = df_3DSC['tc'].values

# Remove outliers
outliers_smix = np.argwhere(np.isnan(smix))
X = np.delete(X, outliers_smix[:, 0], 0)
y = np.delete(yy, outliers_smix[:, 0], 0)

# FEATURE SELECTION
X = pd.DataFrame(X)
corr_features = tsfel.correlated_features(X)
X.drop(corr_features, axis=1, inplace=True)

# Remove low variance features
selector = VarianceThreshold()
X = selector.fit_transform(X)

# NORMALIZATION
scaler = MinMaxScaler(feature_range=(0, 1))
Xn = scaler.fit_transform(X)
yn = scaler.fit_transform(y.reshape(-1, 1))

# EXTRACT REAL Y_TEST
Xn_test = Xn[0:5772, :]
yn_test = yn[0:5772]

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(Xn_test, yn_test, test_size=0.15, random_state=15, shuffle=True)

# # XGBoost
# regr = xgboost.XGBRegressor(
#     n_estimators=1000,  # 800,
#     max_depth=16,  # 7
#     eta=0.02,  # 0.1
#     subsample=1,  # 0.7
#     colsample_bytree=0.5,  # 0.8
#     # booster='dart'
# ).fit(X_train, y_train)

# # PREDICTION
# y_pred = regr.predict(X_test)
# y_pred = y_pred.reshape(-1, 1)
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)

# # Replace zeros and negative values in y_pred with a small non-zero value
# small_value = 1e-10
# y_pred[y_pred <= 0] = small_value

# # PERFORMANCE METRICS
# mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
# mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
# msle = sklearn.metrics.mean_squared_log_error(y_test, y_pred)
# rmse = math.sqrt(mse)
# r2 = sklearn.metrics.r2_score(y_test, y_pred)

# # PRINT METRICS
# print("\nMean absolute error (MAE):      %f" % mae)
# print("Mean squared error (MSE):       %f" % mse)
# print("Mean squared logarithmic error (MSLE): %f" % msle)
# print("Root mean squared error (RMSE): %f" % rmse)
# print("R square (R^2):                 %f" % r2)

# # PLOT RESULTS
# fig, ax = plt.subplots()
# ax.scatter(y_test, y_pred, s=0.4, color='k', zorder=2)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=3, zorder=1)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# plt.show()

# Plotting average number of itinerant electrons per atom (e/a) against Tc
plt.figure(figsize=(10, 6))
# plt.scatter(lattice_vector_a/lattice_vector_c, vecs[:, 9], c=yy, cmap='viridis', edgecolors='k', s=20)
plt.scatter(Maradius, lattice_vector_a/lattice_vector_c, 
            # vecs[:, 9], 
            c=yy, cmap='viridis', #edgecolors='k', 
            s=1)
plt.xlabel('feature x')
plt.ylabel('feature y')
# plt.title('Electronegativity versus Tc')
plt.colorbar(label='Measured TC')
plt.show()