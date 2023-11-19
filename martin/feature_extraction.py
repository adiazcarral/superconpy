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
from keras.layers import (
    Convolution2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    BatchNormalization,
)
import re
from scipy import ndimage, misc
from scipy.stats import entropy
from scipy import signal
from scipy import constants
from scipy.fftpack import fft, dct
from sklearn.preprocessing import MinMaxScaler

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

#####
from vecs import compute_vecs

#####

# class PlotLosses(keras.callbacks.Callback):

#     def on_train_begin(self, logs={}):
#         self.i = 0
#         self.x = []
#         self.losses = []
#         self.val_losses = []

#         self.fig = plt.figure()

#         self.logs = []

#     def on_epoch_end(self, epoch, logs={}):

#         self.logs.append(logs)
#         self.x.append(self.i)
#         self.losses.append(logs.get('loss'))
#         self.val_losses.append(logs.get('val_loss'))
#         self.i += 1

#         clear_output(wait=True)
#         plt.plot(self.x, self.losses, label="loss")
#         plt.plot(self.x, self.val_losses, label="val_loss")
#         plt.title('model loss')
#         plt.legend()
#         #plt.savefig('CNNThy-M-LV-Loss.png')
#         plt.show();
#         # root mean squared error (rmse) for regression
# plot_losses = PlotLosses()


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

    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def r_square_loss(y_true, y_pred):
    from keras import backend as K

    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - (1 - SS_res / (SS_tot + K.epsilon()))


# declare nobel gases
noble_gases = {
    "He": "1s2",
    "Ne": "1s2  2s2 2p6",
    "Ar": " 1s2  2s2 2p6 3s2 3p6",
    "Kr": "1s2  2s2 2p6 3s2 3p6 3d10 4s2 4p6",
    "Xe": "1s2  2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6",
    "Rn": "1s2  2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d10 6s2 6p6",
}


supercon_train = pd.read_csv("/data/angel/Superconductivity/data/unique_m.csv")
supercon_train.head()

supercon_features = supercon_train.copy()
supercon_labels = supercon_features.pop("critical_temp")
supercon_features.drop("material", inplace=True, axis=1)
supercon_features = np.array(supercon_features)
supercon_labels = np.array(supercon_labels)

magpie_train = pd.read_csv("/data/angel/Superconductivity/data/train.csv")
magpie_features = magpie_train.copy()
magpie_features2 = magpie_features
magpie_labels = magpie_features.pop("critical_temp")
magpie_features = np.array(magpie_features)
magpie_labels = np.array(magpie_labels)

zhang_train = pd.read_csv("/data/angel/Superconductivity/data/12340_all.csv")
zhang_features = zhang_train.copy()
zhang_labels = zhang_features.pop("TC")
zhang_features.pop("DOPPED")
zhang_features = np.array(zhang_features)
zhang_labels = np.array(zhang_labels)

formula = pd.read_csv("/work/mroitegui/Superconductors/data/unique_m.csv")
electrones = pd.read_csv(
    "/work/mroitegui/Superconductors/data/periodic_table_of_elementswithelectronstotal.csv"
)
electronegativitycsv = pd.read_csv(
    "/work/mroitegui/Superconductors/data/periodic_table_of_elements.csv"
)
#########
vecs = compute_vecs(formula, electrones, noble_gases)
#########
superconductors_list = formula["material"].tolist()
electroness_dict = dict(zip(electrones["Elemento"], electrones["s"]))
electronesp_dict = dict(zip(electrones["Elemento"], electrones["p"]))
electronesd_dict = dict(zip(electrones["Elemento"], electrones["d"]))
electronesf_dict = dict(zip(electrones["Elemento"], electrones["f"]))
masa_dict = dict(zip(electrones["Elemento"], electrones["Masa"]))
electronegativity_dict = dict(
    zip(electronegativitycsv["Symbol"], electronegativitycsv["Electronegativity"])
)
mend_dict = dict(zip(electrones["elementmend"], electrones["mendnumber"]))
structure_dict = dict(zip(electrones["Elemento"], electrones["Structurenumber"]))
densidadslist = []
densidadplist = []
densidaddlist = []
densidadflist = []
electronesslist = []
electronesplist = []
electronesdlist = []
electronesflist = []
electronegativity_list = []
structure_list = []
mendeleievlist = []
masalist = []
entropy_atvec = np.ndarray(shape=(21263, 1))
shannonvec = np.ndarray(shape=(21263, 1))
renyivec = np.ndarray(shape=(21263, 1))
abevec = np.ndarray(shape=(21263, 1))
kaniadakisvec = np.ndarray(shape=(21263, 1))
psd_atvec = np.ndarray(shape=(21263, 1))
sumatvec = np.ndarray(shape=(21263, 1))


for superconductor in superconductors_list:
    electronesstotal = 0
    electronesptotal = 0
    electronesdtotal = 0
    electronesftotal = 0
    electronegatividadtotal = 0
    atomostotal = 0
    mendtotal = 0
    structure = 0
    masa = 0
    prueba = re.findall(
        "([A-Z][^A-Z]*)", superconductor
    )  # dividimos el superconductor en sus compoentes
    for x in prueba:
        if (
            x.isalpha()
        ):  # Suma un uno al final del elemento en caso de que el numero de atomos no aparezca en la tabla
            x = x + "1"
        elementos = re.sub("[^a-zA-Z]", "", x)
        numeroelectroness = electroness_dict[elementos]
        numeroelectronesp = electronesp_dict[elementos]
        numeroelectronesd = electronesd_dict[elementos]
        numeroelectronesf = electronesf_dict[elementos]
        numeromend = mend_dict[elementos]
        numeromasa = float(masa_dict[elementos])
        numerostruc = structure_dict[elementos]
        electronegatividad = electronegativity_dict[elementos]
        cantidad_de_atomosstr = re.sub("[a-zA-Z]", "", x)
        cantidad_de_atomos = float(cantidad_de_atomosstr)
        cantidadelectroness = cantidad_de_atomos * numeroelectroness
        cantidadelectronesp = cantidad_de_atomos * numeroelectronesp
        cantidadelectronesd = cantidad_de_atomos * numeroelectronesd
        cantidadelectronesf = cantidad_de_atomos * numeroelectronesf
        cantidadelectronegatividad = cantidad_de_atomos * electronegatividad
        cantidadmend = cantidad_de_atomos * numeromend
        cant_struc = numerostruc * cantidad_de_atomos
        electronesstotal += cantidadelectroness
        electronesptotal += cantidadelectronesp
        electronesdtotal += cantidadelectronesd
        electronesftotal += cantidadelectronesf
        structure += cant_struc
        masatotal = cantidad_de_atomos * numeromasa
        masa += masatotal
        atomostotal += cantidad_de_atomos
        mendtotal += cantidadmend
        electronegatividadtotal += cantidadelectronegatividad
    densidads = (electronesstotal) / atomostotal
    densidadslist.append(densidads)
    densidadp = (electronesptotal) / atomostotal
    densidadplist.append(densidadp)
    densidadd = (electronesdtotal) / atomostotal
    densidaddlist.append(densidadd)
    densidadf = (electronesftotal) / atomostotal
    densidadflist.append(densidadf)
    electronesslist.append(numeroelectroness)
    electronesplist.append(numeroelectroness)
    electronesdlist.append(numeroelectroness)
    electronesflist.append(numeroelectroness)
    mean_electronegativity = electronegatividadtotal / atomostotal
    electronegativity_list.append(mean_electronegativity)
    avmend = mendtotal / atomostotal
    mendeleievlist.append(avmend)
    masalist.append(masa)
    structure_list.append(structure)

densidadslist = np.array(densidadslist)
densidadslist = densidadslist.reshape(-1, 1)
densidadplist = np.array(densidadplist)
densidadplist = densidadplist.reshape(-1, 1)
densidaddlist = np.array(densidaddlist)
densidaddlist = densidaddlist.reshape(-1, 1)
densidadflist = np.array(densidadflist)
densidadflist = densidadflist.reshape(-1, 1)
electronesslist = np.array(electronesslist)
electronesslist = electronesslist.reshape(-1, 1)
electronesplist = np.array(electronesplist)
electronesplist = electronesplist.reshape(-1, 1)
electronesdlist = np.array(electronesdlist)
electronesdlist = electronesdlist.reshape(-1, 1)
electronesflist = np.array(electronesflist)
electronesflist = electronesflist.reshape(-1, 1)
eleclist = electronesslist + electronesplist + electronesdlist + electronesflist
electronegativity_list = np.array(electronegativity_list)
electronegativity_list = electronegativity_list.reshape(-1, 1)
mendeleievlist = np.array(mendeleievlist)
mendeleievlist = mendeleievlist.reshape(-1, 1)
masalist = np.array(masalist)
masalist = masalist.reshape(-1, 1)
structure_list = np.array(structure_list)
structure_list = structure_list.reshape(-1, 1)

scaler1 = MinMaxScaler(feature_range=(1, 255))
shannonfeat = supercon_features
# fit scaler on data
scaler1.fit(shannonfeat)
shannonfeat = scaler1.transform(shannonfeat)

for i in range(0, 21263):
    # entropy_atvec[i,:] = entropy(supercon_features[i,:],base=2)
    # supercon_featuresdct = dct(supercon_features[i,:], 1)
    entropy_atvec[i, :] = entropy(supercon_features[i, :])
    sumatvec[i, :] = np.sum(supercon_features[i, :])
    # freqs, psd = signal.welch(supercon_features[i,:])
    # psd_atvec[i,:] = np.max(psd)
    # lap_entropy_atvec = entropy(ndimage.laplace(supercon_features[i,:]),base=2)

# for i in range(0, 21263):
#     s = shannonfeat[i,:]
#     rho = s / s.sum()
#     Slocal = np.sum(rho*np.log2(rho))
#     shannonvec[i] = Slocal

for i in range(0, 21263):
    q = 5  # 5 best for renyi
    k = 0.8
    s = shannonfeat[i, :]
    rho = s / s.sum()
    Slocal1 = (1 / (q - 1)) * np.log(np.sum(rho ** q))
    Slocal2 = -np.sum((rho ** q - rho ** (q ** (-1))) / (q - q ** (-1)))
    Slocal3 = np.sum((rho ** (1 + k) - rho ** (1 - k)) / (2 * k))
    renyivec[i] = Slocal1
    abevec[i] = Slocal2
    kaniadakisvec[i] = Slocal3

maradius = magpie_features2.pop("mean_atomic_radius")
maradius = np.array(maradius)
maradius = maradius.reshape(-1, 1)
meaf = magpie_features2.pop("mean_ElectronAffinity")
meaf = np.array(meaf)
meaf = meaf.reshape(-1, 1)
mtc = magpie_features2.pop("mean_ThermalConductivity")
mtc = np.array(mtc)
mtc = mtc.reshape(-1, 1)
rtc = magpie_features2.pop("range_ThermalConductivity")
rtc = np.array(rtc)
rtc = rtc.reshape(-1, 1)
mval = magpie_features2.pop("mean_Valence")
mval = np.array(mval)
mval = mval.reshape(-1, 1)
matmas = magpie_features2.pop("mean_atomic_mass")
matmas = np.array(matmas)
matmas = matmas.reshape(-1, 1)
mfie = magpie_features2.pop("mean_fie")
mfie = np.array(mfie)
mfie = mfie.reshape(-1, 1)
mdens = magpie_features2.pop("mean_Density")
mdens = np.array(mdens)
mdens = mdens.reshape(-1, 1)
mfh = magpie_features2.pop("mean_FusionHeat")
mfh = np.array(mfh)
mfh = mfh.reshape(-1, 1)
####
adsradius = np.pi * maradius ** 2
adsradius = adsradius.reshape(-1, 1)
adsvol = (4 * np.pi * maradius ** 3) / 3
adsvol = adsvol.reshape(-1, 1)
gibbs = (maradius / 2) - (np.pi * maradius ** 2) * (1 / 4 * np.pi * maradius)
gibbs = gibbs.reshape(-1, 1)
nagasawa = (
    (1 / constants.Boltzmann)
    * (((constants.Planck) ** 2) / (2 * np.pi * 2 * matmas))
    * 2
    * np.pi
    * (1 / (maradius) ** 2)
    * ((2 * eleclist) / (np.log(2 * eleclist)))
)
nagasawa = nagasawa.reshape(-1, 1)
####
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

X = np.concatenate(
    (
        vecs[
            :,
            [0, 4, 5, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        ],
        # densidadslist, densidadplist, densidaddlist, densidadflist,
        # renyivec, #entropy_atvec, kaniadakisvec, abevec, renyivec,
        # electronegativity_list,
        # mendeleievlist,
        # masalist,
        # mval,
        # maradius,
        # mtc
    ),
    axis=1,
)
np.shape(X)
# X = np.concatenate((densidadslist, densidadplist, densidaddlist, densidadflist, entropy_atvec, electronegativity_list,
#     maradius, meaf, mtc, mval, matmas, mfie, mdens, mfh,adsradius, adsvol, gibbs, masalist, mendeleievlist, structure_list), axis= 1)

# X = np.concatenate((maradius, meaf, mtc, mval, matmas, mfie, mdens, mfh), axis= 1) # 84%


####
# y = zhang_labels
y = supercon_labels
# y = magpie_labels
####
# from sklearn.preprocessing import Normalizer
# transformer = Normalizer().fit(X)
# transformer.transform(X)
# # transformer.transform(zhang_labels)

#######
#######
# NORMALIZATION
#######
scaler = MinMaxScaler(feature_range=(0, 1))
# load data
data1 = X
data2 = y
data2 = data2.reshape(-1, 1)
# create scaler
scaler = MinMaxScaler()
#########################
# fit scaler on data
scaler.fit(data1)
Xn = scaler.transform(data1)
scaler.fit(data2)
yn = scaler.transform(data2)
######

X_train, X_test, y_train, y_test = train_test_split(
    Xn, yn, test_size=0.15, random_state=30, shuffle=True
)

# trainn_x, test_x, trainn_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=30, shuffle=True)
# train_x, cval_x, train_y, cval_y = train_test_split(trainn_x, trainn_y, test_size=0.25, random_state=30, shuffle=True)

# ################
# # DEEP LEARNING #
# ################
# from keras.regularizers import l2
# actfun = 'relu'
# reg = l2(0.000001)
# # Training a model

# regressor = Sequential()
# regressor.add(Dense(units=50, activation=actfun, activity_regularizer=reg))
# regressor.add(Dense(units=50, activation=actfun, activity_regularizer=reg))
# regressor.add(Dense(units=100, activation=actfun, activity_regularizer=reg))
# regressor.add(Dense(units=100, activation=actfun, activity_regularizer=reg))
# regressor.add(Dense(units=50, activation=actfun, activity_regularizer=reg))
# regressor.add(Dense(units=50, activation=actfun, activity_regularizer=reg))
# regressor.add(Dense(units=20, activation=actfun, activity_regularizer=reg))
# regressor.add(Dense(units=10, activation=actfun, activity_regularizer=reg))
# regressor.add(Dense(units=1))
# regressor.compile(optimizer='Nadam', loss='mean_squared_error',  metrics=["mean_squared_error", rmse, r_square])

# results=regressor.fit(X_train,y_train, batch_size=700,epochs=800,shuffle=True, validation_data=(X_test,y_test))

# regressor.evaluate(X_test, y_test)


# y_pred = regressor.predict(X_test)

# y_predn= regressor.predict_on_batch(X_test)
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

# cnn = Sequential()
# cnn.add(Conv2D(20, (3, 1),
#     border_mode="same",
#     activation=actfun, activity_regularizer=reg,
#     input_shape=(1, nb_features, 1)))
# cnn.add(Conv2D(20, (3, 1), border_mode="same", activation=actfun, activity_regularizer=reg))
# #cnn.add(BatchNormalization()), use_bias=False
# #cnn.add(MaxPooling2D(pool_size=(1,1)))
# #cnn.add(Dropout(0.5))

# # cnn.add(Conv2D(20, 3, 1, border_mode="same", activation="tanh", activity_regularizer=reg))
# # cnn.add(Conv2D(50, 3, 1, border_mode="same", activation="tanh", activity_regularizer=reg))
# # cnn.add(Dropout(0.5))
# # cnn.add(Conv2D(50, 3, 1, border_mode="same", activation="tanh", activity_regularizer=reg))
# # cnn.add(Conv2D(20, 3, 1, border_mode="same", activation="tanh", activity_regularizer=reg))
# # cnn.add(Convolution2D(100, 3, 1, border_mode="same", activation="tanh", activity_regularizer=reg))
# # cnn.add(Convolution2D(128, 3, 1, border_mode="same", activation="tanh", activity_regularizer=reg))
# #cnn.add(MaxPooling2D(pool_size=(1,1)))
# #
# # cnn.add(Convolution2D(20, 3, 1, border_mode="same", activation="relu", activity_regularizer=reg))
# # cnn.add(Convolution2D(10, 3, 1, border_mode="same", activation="relu", activity_regularizer=reg))
# #cnn.add(Convolution2D(256, 3, 1, border_mode="same", activation="relu"))
# #cnn.add(MaxPooling2D(pool_size=(1,1)))
# cnn.add(Flatten())
# cnn.add(Dense(8, activation=actfun))
# #cnn.add(Dropout(0.5))
# cnn.add(Dense(1, activation=actfun))

# # define optimizer and objective, compile cnn

# cnn.compile(optimizer='Nadam', loss='mean_squared_error',  metrics=["mean_squared_error", rmse, r_square])

# # train


# history = cnn.fit(X_train, y_train, epochs=1500, batch_size=500,
#                   shuffle=True, #callbacks=[plot_losses],
#                   validation_data=(X_test,y_test)
#                   )

# y_pred = cnn.predict(X_test)
# # y_pred2 = np_utils.to_categorical(y_pred)
# #y_pred = np_utils.to_categorical(y_pred)
# # loss, acc = cnn.evaluate(X_test, y_test, verbose=0)
# #print('Test loss:', loss)
# # print('Test accuracy:', acc)
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)
# # ###################

# fig, ax = plt.subplots()
# ax.scatter(y_test, y_pred, s=0.4, color = 'k', zorder=2)
# # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=3  , zorder=1)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# # plt.ylim(-200, 200)
# plt.show()

# ############
# ###MACHINE LEARNING ##
# ###############
# ###########
# ########################
# from sklearn.neighbors import KNeighborsRegressor
# ######
# #k-NN Regressor
# ######
# knn = KNeighborsRegressor(n_neighbors = 4, weights='distance', algorithm= 'auto', p=1)
# y_pred = knn.fit(X_train, y_train).predict(X_test)
# print(knn.score(X_train, y_train))
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)

# # ################
# ## XGBoost
# ##################
# import xgboost
# modelxgb = xgboost.XGBRegressor(n_estimators=2000,
#                                 max_depth=7, eta=0.1,
#                                 subsample=0.7,
#                                 colsample_bytree=0.8).fit(X_train, y_train)
# y_pred = modelxgb.predict(X_test)
# y_pred = y_pred.reshape(-1,1)
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)
################
## Bagging Regressor
################
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor

regr = BaggingRegressor(
    base_estimator=KNeighborsRegressor(
        n_neighbors=2, weights="distance", algorithm="auto", p=1
    ),
    n_estimators=40,
    random_state=0,
).fit(X_train, y_train.ravel())
y_pred = regr.predict(X_test)
y_pred = y_pred.reshape(-1, 1)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)
print(regr.score(X_train, y_train))
###################
# ###################
# # Gaussian Process Regresion #
# ###################
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
# # kernel = DotProduct() + WhiteKernel()
# kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
# # kernel = np.var(yn) * RBF(length_scale=1.0)
# # gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X_train, y_train.ravel())
# gpr = GaussianProcessRegressor(kernel,
#                                   # optimizer=None,
#                                   copy_X_train=False)
# gpr.fit(X_train, y_train.ravel())
# y_pred = gpr.predict(X_test)
# y_pred = np.array(y_pred)
# y_pred = y_pred.reshape(-1,1)
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)
# print(gpr.score(X_train, y_train))
##################
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, s=0.4, color="k", zorder=2)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.plot(
    [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r-", lw=3, zorder=1
)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
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

import sklearn.metrics, math

print("\n")
print(
    "Mean absolute error (MAE):      %f"
    % sklearn.metrics.mean_absolute_error(y_test, y_pred)
)
print(
    "Mean squared error (MSE):       %f"
    % sklearn.metrics.mean_squared_error(y_test, y_pred)
)
print(
    "Root mean squared error (RMSE): %f"
    % math.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred))
)
print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test, y_pred))


#####

# vecs = compute_vecs(formula, electrones, noble_gases)


#######
## SHAP
#######
# import shap

# X_train_summary = shap.kmeans(X_train, 10)
# ex = shap.KernelExplainer(knn.predict, X_train_summary)
# # shap_values = ex.shap_values(X_test.iloc[0,:])
# # shap.force_plot(ex.expected_value, shap_values, X_test.iloc[0,:])
# shap_values = ex.shap_values(X_test)
# shap.summary_plot(shap_values, X_test)
# # shap.dependence_plot("bmi", shap_values, X_test)
# shap.force_plot(ex.expected_value, shap_values, X_test)


# # explain the model's predictions using SHAP
# # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
# explainer = shap.Explainer(modelxgb)
# shap_values = explainer(X)

# # visualize the first prediction's explanation
# shap.plots.waterfall(shap_values[0])
