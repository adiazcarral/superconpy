#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:59:20 2022

@author: angel
"""
import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import tsfel
from sklearn.decomposition import PCA, KernelPCA
import seaborn as sb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalAveragePooling1D, MaxPooling1D, Activation
from keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
import re
from scipy import ndimage, misc
from scipy.stats import entropy
from scipy import signal
from scipy import constants
from scipy.fftpack import fft, dct
from sklearn.preprocessing import MinMaxScaler
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
#####
from vecsz import compute_vecs
#####
#####
import featurefunctions2 as F
#####
import tensorflow as tf
from tensorflow.keras import layers

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

#declare nobel gases
noble_gases = {'He': '1s2',
'Ne':  '1s2  2s2 2p6',
'Ar': ' 1s2  2s2 2p6 3s2 3p6',
'Kr': '1s2  2s2 2p6 3s2 3p6 3d10 4s2 4p6', 
'Xe' :'1s2  2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6',
'Rn' :'1s2  2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d10 6s2 6p6'}


formula = pd.read_csv("/data/angel/Superconductivity/data/12340_all.csv")
zhang_train = pd.read_csv("/data/angel/Superconductivity/data/12340_all.csv")
zhang_features = zhang_train.copy()
zhang_features.pop('DOPPED')
zhang_labels = zhang_features.pop('TC')
zhang_features = np.array(zhang_features)
zhang_labels = np.array(zhang_labels)
electrones=pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elementswithelectronstotal.csv")
electronegativitycsv =pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elements.csv")
#########
vecsz = compute_vecs(formula, electrones, noble_gases)
#########
superconductors_list=formula['DOPPED'].tolist()
electroness_dict=dict(zip(electrones['Elemento'],electrones['s']))
electronesp_dict=dict(zip(electrones['Elemento'],electrones['p']))
electronesd_dict=dict(zip(electrones['Elemento'],electrones['d']))
electronesf_dict=dict(zip(electrones['Elemento'],electrones['f']))
masa_dict=dict(zip(electrones['Elemento'],electrones['Masa']))
electronegativity_dict = dict(zip(electronegativitycsv['Symbol'],electronegativitycsv['Electronegativity']))
mend_dict=dict(zip(electrones['elementmend'],electrones['mendnumber']))
structure_dict=dict(zip(electrones['Elemento'],electrones['Structurenumber']))
densidadslist=[]
densidadplist=[]
densidaddlist=[]
densidadflist=[]
electronesslist=[]
electronesplist=[]
electronesdlist=[]
electronesflist=[]
electronegativity_list=[]
structure_list=[]
mendeleievlist=[]
masalist=[]
# size_dataset = 12376
size_dataset = 12398
entropy_atvec=np.ndarray(shape=(size_dataset,1))
shannonvec=np.ndarray(shape=(size_dataset,1))
renyivec=np.ndarray(shape=(size_dataset,1))
abevec=np.ndarray(shape=(size_dataset,1))
kaniadakisvec=np.ndarray(shape=(size_dataset,1))
psd_atvec=np.ndarray(shape=(size_dataset,1))
sumatvec=np.ndarray(shape=(size_dataset,1))
natoms=np.ndarray(shape=(size_dataset,1))
cnt = 0
for superconductor in superconductors_list:
        electronesstotal=0
        electronesptotal=0
        electronesdtotal=0
        electronesftotal=0
        electronegatividadtotal=0
        atomostotal=0
        mendtotal=0
        structure=0
        masa=0
        prueba=re.findall("([A-Z][^A-Z]*)",superconductor)#dividimos el superconductor en sus compoentes
        for x in prueba:
                    if x.isalpha():#Suma un uno al final del elemento en caso de que el numero de atomos no aparezca en la tabla 
                            x=x+'1'
                    elementos=re.sub('[^a-zA-Z]','',x)
                    numeroelectroness=(electroness_dict[elementos])
                    numeroelectronesp=(electronesp_dict[elementos])
                    numeroelectronesd=(electronesd_dict[elementos])
                    numeroelectronesf=(electronesf_dict[elementos])
                    numeromend=(mend_dict[elementos])
                    numeromasa=float(masa_dict[elementos])
                    numerostruc=(structure_dict[elementos])
                    electronegatividad=(electronegativity_dict[elementos])
                    cantidad_de_atomosstr=re.sub('[a-zA-Z]','',x)
                    cantidad_de_atomos=float(cantidad_de_atomosstr)
                    # print(cantidad_de_atomos)
                    cantidadelectroness=cantidad_de_atomos*numeroelectroness
                    cantidadelectronesp=cantidad_de_atomos*numeroelectronesp
                    cantidadelectronesd=cantidad_de_atomos*numeroelectronesd
                    cantidadelectronesf=cantidad_de_atomos*numeroelectronesf
                    cantidadelectronegatividad=cantidad_de_atomos*electronegatividad
                    cantidadmend=cantidad_de_atomos*numeromend
                    cant_struc=numerostruc*cantidad_de_atomos
                    electronesstotal+=cantidadelectroness
                    electronesptotal+=cantidadelectronesp
                    electronesdtotal+=cantidadelectronesd
                    electronesftotal+=cantidadelectronesf
                    structure+=cant_struc
                    masatotal=cantidad_de_atomos*numeromasa
                    masa+=masatotal
                    atomostotal+=cantidad_de_atomos
                    mendtotal+=cantidadmend
                    electronegatividadtotal+=cantidadelectronegatividad
        natoms[cnt] = atomostotal
        cnt+=1
        densidads=(electronesstotal)/atomostotal
        densidadslist.append(densidads)
        densidadp=(electronesptotal)/atomostotal
        densidadplist.append(densidadp)
        densidadd=(electronesdtotal)/atomostotal
        densidaddlist.append(densidadd)
        densidadf=(electronesftotal)/atomostotal
        densidadflist.append(densidadf)
        electronesslist.append(numeroelectroness)
        electronesplist.append(numeroelectroness)
        electronesdlist.append(numeroelectroness)
        electronesflist.append(numeroelectroness)
        mean_electronegativity=electronegatividadtotal/atomostotal
        electronegativity_list.append(mean_electronegativity)
        avmend=mendtotal/atomostotal
        mendeleievlist.append(avmend)
        masalist.append(masa)
        structure_list.append(structure)

densidadslist = np.array(densidadslist)
densidadslist = densidadslist.reshape(-1,1)
densidadplist = np.array(densidadplist)
densidadplist = densidadplist.reshape(-1,1)
densidaddlist = np.array(densidaddlist)
densidaddlist = densidaddlist.reshape(-1,1)
densidadflist = np.array(densidadflist)
densidadflist = densidadflist.reshape(-1,1)
electronesslist = np.array(electronesslist)
electronesslist = electronesslist.reshape(-1,1)
electronesplist = np.array(electronesplist)
electronesplist = electronesplist.reshape(-1,1)
electronesdlist = np.array(electronesdlist)
electronesdlist = electronesdlist.reshape(-1,1)
electronesflist = np.array(electronesflist)
electronesflist = electronesflist.reshape(-1,1)
eleclist = electronesslist + electronesplist + electronesdlist + electronesflist
electronegativity_list =np.array(electronegativity_list)
electronegativity_list=electronegativity_list.reshape(-1,1)
mendeleievlist = np.array(mendeleievlist)
mendeleievlist=mendeleievlist.reshape(-1,1)
masalist=np.array(masalist)
masalist=masalist.reshape(-1,1)
structure_list = np.array(structure_list)
structure_list = structure_list.reshape(-1,1)
EN=F.electrodifference()
Mval=F.mval()
Mtc=F.mtc()
Maradius = F.mrad()
Mfie=F.mfie()
Mec=F.mec()
############

####
# TRAINING MATRIX #
#####

# X = supercon_features
# X = magpie_features
# X = zhang_features
# X = sumatvec
# X = np.concatenate((electronesslist, electronesplist, electronesdlist, electronesflist), axis= 1) 
# X = np.concatenate((densidadslist, densidadplist, densidaddlist, densidadflist), axis= 1) #88% prec
# X = np.concatenate((densidadslist, densidadplist, densidaddlist, densidadflist, #entropy_atvec, structure_list, masalist
#                      mval, electronegativity_list, mendeleievlist, maradius, masalist), axis= 1) 

# X = np.concatenate((densidadslist, 
#                     densidadplist, 
#                     # densidaddlist,
#                     densidadflist,
#                     # entropy_atvec, # renyivec, kaniadakisvec, abevec,
#                     # mval, 
#                     electronegativity_list, 
#                     mendeleievlist,
#                     # maradius,
#                     masalist
#                     ), axis= 1) 

X = np.concatenate((
                    vecsz[:,[0,4,5,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25]],
                    # densidadslist, 
                    # densidadplist, 
                    # densidaddlist, 
                    # densidadflist, 
                    # renyivec, #entropy_atvec, kaniadakisvec, abevec, renyivec,
                    # electronegativity_list, 
                    # mendeleievlist, 
                    # masalist,
                    # natoms,
                    # EN,
                    # Maradius,
                    # Mval,
                    # Mtc,
                    # Mfie,
                    # Mec
                    ), axis= 1) 

# X = zhang_features
# X = np.concatenate((densidadslist, densidadplist, densidaddlist, densidadflist, entropy_atvec, electronegativity_list, 
#     maradius, meaf, mtc, mval, matmas, mfie, mdens, mfh,adsradius, adsvol, gibbs, masalist, mendeleievlist, structure_list), axis= 1)

# X = np.concatenate((maradius, meaf, mtc, mval, matmas, mfie, mdens, mfh), axis= 1) # 84%

####
y = zhang_labels
#######
#NORMALIZATION
#######
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
X_test2 = Xn[12344:size_dataset,:]
y_test2 = yn[12344:size_dataset]
Xn = Xn[0:12340, :]
yn = yn[0:12340]
######
##### Natoms scaler
# scaler = MinMaxScaler()
#########################
# # fit scaler on data
# scaler.fit(data1)
# Xn = scaler.transform(data1)
#####
X_train, X_test, y_train, y_test = train_test_split(Xn, yn, test_size=0.15, random_state=30, shuffle=True)

# trainn_x, test_x, trainn_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=30, shuffle=True)
# train_x, cval_x, train_y, cval_y = train_test_split(trainn_x, trainn_y, test_size=0.25, random_state=30, shuffle=True)

# ################
# # DEEP LEARNING #
# ################
# from keras.regularizers import l2
# actfun = 'relu'
# reg = l2(0.000001)
# # Training a model

# regr = Sequential()
# regr.add(Dense(units=50, activation=actfun, activity_regularizer=reg))
# regr.add(Dense(units=50, activation=actfun, activity_regularizer=reg))
# regr.add(Dense(units=100, activation=actfun, activity_regularizer=reg))
# regr.add(Dense(units=100, activation=actfun, activity_regularizer=reg))
# regr.add(Dense(units=50, activation=actfun, activity_regularizer=reg))
# regr.add(Dense(units=50, activation=actfun, activity_regularizer=reg))
# regr.add(Dense(units=20, activation=actfun, activity_regularizer=reg))
# regr.add(Dense(units=10, activation=actfun, activity_regularizer=reg))
# regr.add(Dense(units=1))
# regr.compile(optimizer='Nadam', loss='mean_squared_error',  metrics=["mean_squared_error", rmse, r_square])

# results=regr.fit(X_train,y_train, batch_size=2000,epochs=800,shuffle=True, validation_data=(X_test,y_test))

# regr.evaluate(X_test, y_test)


# y_pred = regr.predict(X_test)

# y_predn= regr.predict_on_batch(X_test)
# y_pred = scaler.inverse_transform(y_predn)
# y_test = scaler.inverse_transform(y_test)


# from sklearn.metrics import mean_squared_error

# # plot training curve for R^2 (beware of scale, starts very low negative)
# plt.plot(results.history['r_square'])
# plt.plot(results.history['val_r_square'])
# plt.title('model R^2')
# plt.ylabel('R^2')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
           
# # plot training curve for rmse
# plt.plot(results.history['rmse'])
# plt.plot(results.history['val_rmse'])
# plt.title('rmse')
# plt.ylabel('rmse')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# ################
# fig, ax = plt.subplots()
# ax.scatter(y_test, y_pred, s=0.4, color = 'k', zorder=2)
# # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=3  , zorder=1)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# # plt.ylim(-200, 200)
# plt.show()
########
########
# ########
# # CNN 
# ####
# from keras.models import Sequential
# from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
# from keras.utils import np_utils
# import numpy as np
# from keras.regularizers import l2
# # instantiate regularizer
# reg = l2(0.000000001)
# #from keras.optimizers import Adam
# training_length = np.shape(X_train)[0]
# test_length = np.shape(X_test)[0]
# nb_features = np.size(X_train[0])
# X_train = X_train.reshape((training_length, 1, nb_features, 1))
# X_test = X_test.reshape((test_length, 1, nb_features, 1))
# test_y11 = y_test

# actfun = 'relu'

# regr = Sequential()
# regr.add(Conv2D(20, (3, 1),
#     border_mode="same",
#     activation=actfun, activity_regularizer=reg,
#     input_shape=(1, nb_features, 1)))
# regr.add(Conv2D(20, (3, 1), border_mode="same", activation=actfun, activity_regularizer=reg))
# #regr.add(BatchNormalization()), use_bias=False
# #regr.add(MaxPooling2D(pool_size=(1,1)))
# #regr.add(Dropout(0.5))

# # regr.add(Conv2D(20, 3, 1, border_mode="same", activation="tanh", activity_regularizer=reg))
# # regr.add(Conv2D(50, 3, 1, border_mode="same", activation="tanh", activity_regularizer=reg))
# # regr.add(Dropout(0.5))
# # regr.add(Conv2D(50, 3, 1, border_mode="same", activation="tanh", activity_regularizer=reg))
# # regr.add(Conv2D(20, 3, 1, border_mode="same", activation="tanh", activity_regularizer=reg))
# # regr.add(Convolution2D(100, 3, 1, border_mode="same", activation="tanh", activity_regularizer=reg))
# # regr.add(Convolution2D(128, 3, 1, border_mode="same", activation="tanh", activity_regularizer=reg))
# #regr.add(MaxPooling2D(pool_size=(1,1)))
# #   
# # regr.add(Convolution2D(20, 3, 1, border_mode="same", activation="relu", activity_regularizer=reg))
# # regr.add(Convolution2D(10, 3, 1, border_mode="same", activation="relu", activity_regularizer=reg))
# #regr.add(Convolution2D(256, 3, 1, border_mode="same", activation="relu"))
# #regr.add(MaxPooling2D(pool_size=(1,1)))
# regr.add(Flatten())
# regr.add(Dense(8, activation=actfun))
# #regr.add(Dropout(0.5))
# regr.add(Dense(1, activation=actfun))

# # define optimizer and objective, compile cnn

# regr.compile(optimizer='Nadam', loss='mean_squared_error',  metrics=["mean_squared_error", rmse, r_square])

# # train


# history = regr.fit(X_train, y_train, epochs=1500, batch_size=500, 
#                   shuffle=True, #callbacks=[plot_losses], 
#                   validation_data=(X_test,y_test)
#                   )

# y_pred = regr.predict(X_test)
# # y_pred2 = np_utils.to_categorical(y_pred)
# #y_pred = np_utils.to_categorical(y_pred)
# # loss, acc = cnn.evaluate(X_test, y_test, verbose=0)
# #print('Test loss:', loss)
# # print('Test accuracy:', acc)
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)
# ###################

# fig, ax = plt.subplots()
# ax.scatter(y_test, y_pred, s=0.4, color = 'k', zorder=2)
# # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=3  , zorder=1)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# # plt.ylim(-200, 200)
# plt.show()

###########
#MACHINE LEARNING ##
#############
# ##############
# from sklearn.neighbors import KNeighborsRegressor
# ######
# #k-NN Regressor
# ######
# regr = KNeighborsRegressor(n_neighbors = 3, weights='distance', algorithm= 'auto', p=1)
# y_pred = regr.fit(X_train, y_train).predict(X_test)
# print(regr.score(X_train, y_train))
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)

# # ################
# ## XGBoost
# ##################
# import xgboost
# modelxgb = xgboost.XGBRegressor(n_estimators=500, 
#                                 max_depth=7, eta=0.1, 
#                                 subsample=0.7, 
#                                 colsample_bytree=0.8).fit(X_train, y_train)
# y_pred = modelxgb.predict(X_test)
# y_pred = y_pred.reshape(-1,1)
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)
# # ################
# ## Bagging Regressor
# ################
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.ensemble import BaggingRegressor
# #####################
# regr = BaggingRegressor(base_estimator=KNeighborsRegressor(n_neighbors = 5, weights='distance', algorithm= 'auto', p=1),
#                         n_estimators=20, ## it was 50
#                         random_state=0).fit(X_train, y_train.ravel())
# y_pred = regr.predict(X_test)
# y_pred = y_pred.reshape(-1,1)
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)
# print(regr.score(X_train,y_train))
# ################
## XGBoost
##################
import xgboost

regr = xgboost.XGBRegressor(n_estimators=1500, # 200
                                max_depth=15, # 7
                                eta=0.01, # 0.1 
                                # gamma = 0.1, #0.5
                                subsample=0.7, # 0.7
                                colsample_bytree=0.8, # 0.8
                                # booster = 'dart'
                                ).fit(X_train, y_train)
# regr = xgboost.XGBRegressor(
#     # n_estimators=500,
#     # eta=0.1
#     ).fit(X_train, y_train)
y_pred = regr.predict(X_test)
y_pred = y_pred.reshape(-1,1)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

#####
import shap
# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(regr)
shap_values = explainer(X_train)
# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])
shap.plots.force(shap_values[0])
# summarize the effects of all the features
shap.plots.beeswarm(shap_values)
shap.plots.bar(shap_values)
#################
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, s=0.4, color = 'k', zorder=2)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=3  , zorder=1)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
# plt.ylim(-200, 200)
plt.show()

# pred_train= regressor.predict(X_train)
# print(np.sqrt(mean_squared_error(y_train,pred_train)))

# pred= regressor.predict(X_test)
# print(np.sqrt(mean_squared_error(y_test,pred))) 

# =============================================================================
# results = np.ndarray(shape=(886,2))
# results[:,0] = test_y
# results[:,1] = y_pred
# scipy.io.savemat('Hpred.mat',mdict={'Hpredd':results})
# =============================================================================
from sklearn.metrics import f1_score
import sklearn.metrics, math
print("\n")
print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test,y_pred))
print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test,y_pred))
print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test,y_pred)))
print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test,y_pred))
###### PREDICTION OF NBSN AND NEW PROTOTYPES #######
y_pred2 = regr.predict(X_test2)
y_pred2 = y_pred2.reshape(-1,1)
y_pred2 = scaler.inverse_transform(y_pred2)
y_test2 = scaler.inverse_transform(y_test2)
y_pred2 = np.around(y_pred2, decimals=2)

print(y_test2.ravel())
print(y_pred2.ravel())
fig, ax = plt.subplots()
ax.scatter(y_test2, y_pred2, s=5, color = 'k', zorder=2)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.plot([y_test2.min(), y_test2.max()], [y_test2.min(), y_test2.max()], 'r-', lw=3  , zorder=1)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
# plt.ylim(-200, 200)
plt.show()
# ### Error distribution ###
# """
# Calculate error from between target and predictions
# (Based on merged data frame of test data and predictions)
# """
# atoms = X_test[:,23]
# atoms = atoms.reshape(-1,1)
# atoms = scaler.inverse_transform(atoms)
# y_error = np.abs(y_test - y_pred)

# #Set plot size
# plt.subplots(figsize=(10,5))
# #Set X-Axis range
# # plt.xlim(0, 180)
# plt.title('Model Error Distribution')
# plt.ylabel('Abs. Error (Kelvin)')
# plt.xlabel('Error')
# plt.scatter(atoms, y_error)
# plt.show()


