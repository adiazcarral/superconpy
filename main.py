#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:58:26 2023

@author: mroitegui
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:47:15 2023

@author: mroitegui
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:59:20 2022

@author: angel
"""
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import tsfel
from sklearn.decomposition import PCA, KernelPCA
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
from sklearn import manifold
from sklearn.manifold import TSNE, MDS
from sklearn.neighbors import KNeighborsRegressor
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
import featurefunctions as F

        
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))

zhang_train = pd.read_csv("data/12340_all_pred.csv")
zhang_features = zhang_train.copy()
zhang_labels = zhang_features.pop('TC')
zhang_features.pop('DOPPED')
zhang_features = np.array(zhang_features)
zhang_labels = np.array(zhang_labels)


# folder_path='/work/mroitegui/Superconductivity/shap'
##########
formula = pd.read_csv("data/12340_all_pred.csv")
electrones=pd.read_csv("data/periodic_table_of_elementswithelectronstotal.csv")
superconductors_list=formula['DOPPED'].tolist()
#########
#########
#FEATURES#
EN=F.electrodifference(formula,electrones,superconductors_list)
Mval=F.mval(formula,electrones,superconductors_list)
Mtc=F.mtc(formula,electrones,superconductors_list)
Maradius = F.mrad(formula,electrones,superconductors_list)
Mfie=F.mfie(formula,electrones,superconductors_list)
Mec=F.mec(formula,electrones,superconductors_list)
Masa=F.masa(formula,electrones,superconductors_list)
densidads,densidadp,densidadd,densidadf=F.eletronicdensity(formula,electrones,superconductors_list)
Mend=F.mend(formula,electrones,superconductors_list)
vecs = F.compute_vecs(formula, electrones,superconductors_list)
electronegativity_list=F.EN_old(formula, electrones, superconductors_list)
smix = F.mixing_entropy(superconductors_list)
Delta = F.delta(electrones, superconductors_list)
############
# outliers_smix = np.argwhere(np.isnan(smix))
# print(outliers_smix[:,0])
# outliers = np.asarray(outliers_smix[:,0])
# formula.drop(outliers,axis=0,inplace=True)
####
# TRAINING MATRIX #
#####
size_dataset = np.size(superconductors_list)

features=[
    # '1s','2s','2p',
           '3s',
           # '3p',
            '3d',
           # '4s',
           # '4p','4d','4f','5s','5p','5d','5f',
            '6s',
           # '6p','6d','6f',
           # '7s',
           # '7p',
                            'Mend', 
                            # 'Mass',
                            # 'EN',
                            # 'Maradius',
                            # 'Mval',
                            # 'Mtc',
                            'Mfie',
                            'Mec',
                            'Smix',
                            # 'delta',
                        
                        ]
X = np.concatenate((
                    vecs[:,[
                        # 0,4,5,
                            8,
                            # 9,
                            10,
                            # 12,
                            # 13,14,15,16,17,18,19,
                            20,
                            # 21,22,23,24,25
                            ]],                  
                    Mend, 
                    # Masa,
                    # EN,
                    # Maradius,
                    # Mval,
                    # Mtc,
                    Mfie,
                    Mec,
                    smix,
                    # Delta,
                    ), axis= 1) 
y = zhang_labels
print(np.shape(X))
# ------------------- #
###################################
# ------- --------------- ------- #
# ------- REMOVE OUTLIERS ------- #
# ------- --------------- ------- #
###################################
outliers_smix = np.argwhere(np.isnan(smix))
# print(outliers_smix[:,0])
outliers = np.asarray(outliers_smix[:,0])
X = np.delete(X, outliers, 0)
y = np.delete(y, outliers, 0)
# ####
# print(np.shape(X))
# print(np.shape(y))
###################################
# ------- --------------- ------- #
# ------- FEATURE SELECTION ----- #
# ------- --------------- ------- #
###################################
# Highly correlated features are removed
X = pd.DataFrame(X)
corr_features = tsfel.correlated_features(X)
X.drop(corr_features, axis=1, inplace=True)
# Remove low variance features
selector = VarianceThreshold()
X = selector.fit_transform(X)
# X = X.to_numpy()
print(np.shape(X))
###################################
# ------- --------------- ------- #
# ------- NORMALIZATION-- ------- #
# ------- --------------- ------- #
###################################
scaler = MinMaxScaler(feature_range=(0,1))
# load data
data1 = X
data2 = y
data2 = data2.reshape(-1, 1)
# create scaler
# scaler = MinMaxScaler()
#########################
# fit scaler on data
scaler.fit(data1)
Xn = scaler.transform(data1)
scaler.fit(data2)
yn = scaler.transform(data2)
######
#######
## EXTRACT REAL Y_TEST ##
#######
# ####ZHANG#####
# X_test2 = Xn[12340:size_dataset,:]
# y_test2 = yn[12340:size_dataset]
Xn = Xn[0:size_dataset, :]
yn = yn[0:size_dataset]
###########

X_train, X_test, y_train, y_test = train_test_split(Xn, yn, test_size=0.15, random_state=30, shuffle=True)

# ############
# ##MACHINE LEARNING ##
# ###########
# # ----- k-NN Regressor -----#
# ###########
# regr = KNeighborsRegressor(n_neighbors = 10, 
#                             weights='distance', # weights='distance', 
#                             algorithm= 'auto', 
#                             p=1
#                             ).fit(X_train, y_train)

# y_pred = regr.predict(X_test)
# y_pred = y_pred.reshape(-1,1)
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)
# # # ################
# ## Bagging Regressor
# ################
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.ensemble import BaggingRegressor

# regr = BaggingRegressor(estimator=KNeighborsRegressor(n_neighbors = 3, 
#                                                             weights='distance', 
#                                                             algorithm= 'auto', 
#                                                             p=1,
#                                                             # n_estimators=5, ## it was 50
#                                                             # random_state=0
#                                                             )).fit(X_train, y_train.ravel())
# y_pred = regr.predict(X_test)
# y_pred = y_pred.reshape(-1,1)
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)
# print(regr.score(X_train,y_train))
# # ################
# # # Random Forest
# # ################
# from sklearn.ensemble import RandomForestRegressor
# regr = RandomForestRegressor()
# regr.fit(X_train, y_train.ravel())
# y_pred = regr.predict(X_test)
# y_pred = y_pred.reshape(-1,1)
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)
################
# XGBoost
################
import xgboost
regr = xgboost.XGBRegressor(n_estimators= 1000,  #800, 
                                max_depth=16, # 7
                                eta=0.02, # 0.1 
                                subsample=1, # 0.7
                                colsample_bytree=0.5, # 0.8
                                # booster = 'dart'
                                ).fit(X_train, y_train)
y_pred = regr.predict(X_test)
y_pred = y_pred.reshape(-1,1)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)
###########################

import sklearn.metrics, math
print("\n")
print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test,y_pred))
print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test,y_pred))
print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test,y_pred)))
print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test,y_pred))

MAE=np.absolute(y_test-y_pred)
MSE= np.square(y_test-y_pred)
RMSE=np.sqrt(MSE)

# # #################
#Plot results
#####################
####Normal
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, s=0.4, color = 'k', zorder=2)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=3  , zorder=1)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
# plt.ylim(-200, 200)

# ######Boxplot
# fig, ax = plt.subplots()
# # ax.boxplot(MAE, 0, 'gD')
# ax.boxplot(MAE)
# ax.set_xlabel('MAE')
# plt.show()

# fig, ax = plt.subplots()
# ax.boxplot(MSE)
# ax.set_xlabel('MSE')
# plt.show()
# ##################################
# ------- --------------- ------- #
# ------- SHAPLEY VALUES- ------- #
# ------- --------------- ------- #
# ##################################
import shap
import time
features.remove('4s')
features.remove('6d')
features.remove('7s')
X_train=pd.DataFrame(X_train,columns = features)
# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
### knn explainer ###
# t0 = time.time()
# X_train_summary = shap.kmeans(X_train, 10)
# explainerKNN = shap.KernelExplainer(regr.predict, X_train_summary)
# shap_values = explainerKNN(X_train_summary)
# t1 = time.time()
# timeit=t1-t0
# timeit
### XGBoost explainer ###
# explainer = shap.Explainer(regr)
# shap_values = explainer(X_train)
# ############
# ### RF explainer ###
# # Create object that can calculate shap values
# explainer = shap.TreeExplainer(regr)
# # Calculate Shap values
# shap_values = explainer.shap_values(X_train)
# #################
# shap_df=pd.DataFrame(shap_values.values, columns=features)
# shap_df=shap_df.abs()
# shap_df_mean=shap_df.mean()
# shap_vecs=shap_df_mean[['1s','2s','2p','3s','3p','3d',
#                         # '4s',
#                         '4p','4d','4f','5s','5p','5d','5f','6s','6p',
#                         # '6d',
#                         '6f',
#                         # '7s',
#                         '7p']].copy()
# shap_df_mean['vec']=shap_df_mean[:20].sum()
# shap_total=shap_df_mean.copy()
# shap_total=shap_total.drop(['1s','2s','2p','3s','3p','3d',
#                             # '4s',
#                             '4p','4d','4f','5s','5p','5d','5f','6s','6p',
#                             # '6d',
#                             '6f',
#                             # '7s',
#                             '7p'])

# shap_total.plot(ylabel='Mean Shap values',kind='bar')
# plt.show()

# shap_vecs.plot(ylabel='Mean Shap values',kind='bar')
# plt.show()

###################
# END OF THE CODE #
###################
