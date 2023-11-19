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
from sklearn import manifold
from sklearn.manifold import TSNE, MDS
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
#####
from vecs import compute_vecs
#####
import featurefunctions as F
#####
import tensorflow as tf
from tensorflow.keras import layers

class PlotLosses(keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.title('model loss')
        plt.legend()
        #plt.savefig('CNNThy-M-LV-Loss.png')
        plt.show();
        # root mean squared error (rmse) for regression
plot_losses = PlotLosses()        
        
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

stanev_train= pd.read_csv('/work/mroitegui/Superconductors/data/Supercon_data_stanev.csv')
supercon_train = pd.read_csv("/work/angel/Superconductivity/datasets/archive/unique_m.csv")
supercon_train.head()

supercon_features = supercon_train.copy()
supercon_labels = supercon_features.pop('critical_temp')
supercon_features.drop('material', inplace=True, axis=1)
supercon_features = np.array(supercon_features)
supercon_labels = np.array(supercon_labels)

magpie_train = pd.read_csv("/data/angel/Superconductivity/data/train.csv")
magpie_features = magpie_train.copy()
magpie_features2 = magpie_features
magpie_labels = magpie_features.pop('critical_temp')
magpie_features = np.array(magpie_features)
magpie_labels = np.array(magpie_labels)

zhang_train = pd.read_csv("/work/mroitegui/Superconductors/data/12340_all_pred.csv")
zhang_features = zhang_train.copy()
zhang_labels = zhang_features.pop('TC')
zhang_features.pop('DOPPED')
zhang_features = np.array(zhang_features)
zhang_labels = np.array(zhang_labels)

stanev_features = stanev_train.copy()
stanev_labels = stanev_features.pop('Tc')
stanev_features.drop('name', inplace=True, axis=1)
stanev_features = np.array(stanev_features)
stanev_labels = np.array(stanev_labels)


##########
# formula = pd.read_csv("/work/mroitegui/Superconductors/data/12340_all_pred.csv")
formula = pd.read_csv("/work/angel/Superconductivity/datasets/archive/unique_m.csv")
# electrones=pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elements.csv")
electrones=pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elementswithelectronstotal.csv")
# electronegativitycsv =pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elements.csv")
superconductors_list=formula['material'].tolist()
# superconductors_list=formula['DOPPED'].tolist()
#########

#########
#FEATURES#
# size_dataset = 12376
size_dataset=21299
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
                    vecs[:,[0,4,5,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25]],
                    # vecs[:,[
                    #         0,    # 0 
                    #         4, 5, # 4, 5,
                    #         9, 10, # 8, 9, 10,
                    #         12, 13, 15,# 12 ,13 ,14, 15, 
                    #         # 16, 17, 19,# 16, 17, 18, 19,
                    #         20, 21, 23, # 20, 21, 22, 23,
                    #         # 24, 25 # 24, 25
                    #         ]], 
                    # densidads, 
                    # densidadp, 
                    # densidadd, 
                    # densidadf, 
                    # # renyivec, entropy_atvec, kaniadakisvec, abevec, renyivec,
                    ## electronegativity_list, 
                    Mend, 
                    Masa,
                    EN,
                    Maradius,
                    Mval,
                    Mtc,
                    Mfie,
                    Mec
                    # # # natoms,
                  
                    ), axis= 1) 

print(np.shape(X))
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
# ####ZHANG#####
# X_test2 = Xn[12340:size_dataset,:]
# y_test2 = yn[12340:size_dataset]
# Xn = Xn[0:12340, :]
# yn = yn[0:12340]
###########
#####Supercon##########
X_test2 = Xn[21263:size_dataset,:]
y_test2 = yn[21263:size_dataset]
Xn = Xn[0:21263, :]
yn = yn[0:21263]
######
#####Supercon_Stanev##########
# X_test2 = Xn[16415:size_dataset,:]
# y_test2 = yn[16415:size_dataset]
# Xn = Xn[0:16415, :]
# yn = yn[0:16415]
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
# reg = l2(0.0000001)
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

# results=regr.fit(X_train,y_train, batch_size=800,epochs=500,shuffle=True, validation_data=(X_test,y_test))

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

################
# fig, ax = plt.subplots()
# ax.scatter(y_test, y_pred, s=0.4, color = 'k', zorder=2)
# # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=3  , zorder=1)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# # plt.ylim(-200, 200)
# plt.show()
#####
#######
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
#     # border_mode="same",
#     activation=actfun, activity_regularizer=reg,
#     input_shape=(1, nb_features, 1)))
# regr.add(Conv2D(20, (3, 1), 
#                 # border_mode="same", 
#                 activation=actfun, activity_regularizer=reg))
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


# history = regr.fit(X_train, y_train, epochs=500, batch_size=500, 
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

# ############
# ##MACHINE LEARNING ##
# ##############
# ##########
# #######################
# from sklearn.neighbors import KNeighborsRegressor
######
#k-NN Regressor
######
# regr = KNeighborsRegressor(n_neighbors = 3, 
#                             weights='distance', # weights='distance', 
#                             algorithm= 'auto', 
#                             p=1
#                             )
# y_pred = regr.fit(X_train, y_train).predict(X_test)
# print(regr.score(X_train, y_train))
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)

# ################
# # XGBoost
# ################
# import xgboost
# regr = xgboost.XGBRegressor(n_estimators=800, # 200
#                                 max_depth=16, # 7
#                                 eta=0.02, # 0.1 
#                                 subsample=1, # 0.7
#                                 colsample_bytree=0.5, # 0.8
#                                 # booster = 'dart'
#                                 ).fit(X_train, y_train)
# y_pred = regr.predict(X_test)
# y_pred = y_pred.reshape(-1,1)
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)
# X_train=pd.DataFrame(X_train,columns =[
#                         '1s','2s','2p','3s','3p','3d','4s','4p','4d','4f','5s','5p','5d','5f','6s','6p','6d','6f','7s','7p',
#                         # 'densidads', 
#                         # 'densidadp', 
#                         # 'densidadd', 
#                         # 'densidadf', 
#                        # renyivec, entropy_atvec, kaniadakisvec, abevec, renyivec,
#                        # ' electronegativity_list', 
#                             'Mend', 
#                             'Masa',
#                             'EN',
#                             'Maradius',
#                             'Mval',
#                             'Mtc',
#                             'Mfie',
#                                 'Mec'
#                         # # # # # natoms,
                        
#                        ]
#                      )
# ############
# import shap
# # explain the model's predictions using SHAP
# # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
# explainer = shap.Explainer(regr)
# shap_values = explainer(X_train)
# # visualize the first prediction's explanation
# shap.plots.waterfall(shap_values[0],max_display=27)
# shap.plots.force(shap_values[0])
# # summarize the effects of all the features
# shap.plots.beeswarm(shap_values,max_display=27)
# shap.plots.bar(shap_values,max_display=27)
# #################
# fig, ax = plt.subplots()
# ax.scatter(y_test, y_pred, s=0.4, color = 'k', zorder=2)
# # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=3  , zorder=1)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# # plt.ylim(-200, 200)
# plt.show()


# # # ################
# ## Bagging Regressor
# ################
# # from sklearn.neighbors import KNeighborsRegressor
# # from sklearn.ensemble import BaggingRegressor

# # regr = BaggingRegressor(base_estimator=KNeighborsRegressor(n_neighbors = 3, 
# #                                                             weights='distance', 
# #                                                             algorithm= 'auto', 
# #                                                             p=1),
# #                                                             n_estimators=5, ## it was 50
# #                                                             random_state=0).fit(X_train, y_train.ravel())
# # y_pred = regr.predict(X_test)
# # y_pred = y_pred.reshape(-1,1)
# # y_pred = scaler.inverse_transform(y_pred)
# # y_test = scaler.inverse_transform(y_test)
# # print(regr.score(X_train,y_train))
# # # ###################
# # #################
# # fig, ax = plt.subplots()
# # ax.scatter(y_test, y_pred, s=0.4, color = 'k', zorder=2)
# # # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=3  , zorder=1)
# # ax.set_xlabel('Measured')
# # ax.set_ylabel('Predicted')
# # # plt.ylim(-200, 200)
# # plt.show()

# # pred_train= regressor.predict(X_train)
# # print(np.sqrt(mean_squared_error(y_train,pred_train)))

# # pred= regressor.predict(X_test)
# # print(np.sqrt(mean_squared_error(y_test,pred))) 
# # fig, ax = plt.subplots()
# # ax.scatter(y_test, y_pred, s=0.4, color="k", zorder=2)
# # # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# # ax.plot(
# #     [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r-", lw=3, zorder=1
# # )
# # ax.set_xlabel("Measured")
# # ax.set_ylabel("Predicted")
# # # plt.ylim(-200, 200)
# # plt.show()
# # =============================================================================
# # results = np.ndarray(shape=(886,2))
# # results[:,0] = test_y
# # results[:,1] = y_pred
# # scipy.io.savemat('Hpred.mat',mdict={'Hpredd':results})
# # =============================================================================
# import sklearn.metrics, math
# print("\n")
# print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test,y_pred))
# print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test,y_pred))
# print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test,y_pred)))
# print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test,y_pred))



# ###### PREDICTION OF NBSN AND NEW PROTOTYPES #######
# y_pred2 = regr.predict(X_test2)
# y_pred2 = y_pred2.reshape(-1,1)
# y_pred2 = scaler.inverse_transform(y_pred2)
# y_test2 = scaler.inverse_transform(y_test2)
# y_pred2 = np.around(y_pred2, decimals=2)

# print('Test')
# print(y_test2.ravel())
# print('Pred')
# print(y_pred2.ravel())
# fig, ax = plt.subplots()
# ax.scatter(y_test2, y_pred2, s=5, color = 'k', zorder=2)
# # ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
# ax.plot([y_test2.min(), y_test2.max()], [y_test2.min(), y_test2.max()], 'r-', lw=3  , zorder=1)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# # plt.ylim(-200, 200)
# plt.show()
# # ### Error distribution ###
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

################
# CLUSTERING ###
################
data = Xn
################
## Kernel PCA 2D##
################
##  kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’}, default=’linear’
kernell = "linear"
s = 0.05
color = 'b'
transformer = KernelPCA(n_components=2, 
                        kernel=kernell,
                        gamma=None) 
                        #, fit_inverse_transform=True, alpha=0.1)
# X_transformed = transformer.fit_transform(X_train)
X_transformed = transformer.fit_transform(data)
print(X_transformed.shape)

###############
## plot results
###############
fig, ax = plt.subplots()
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], s=s, color = color, zorder=2)
ax.set_ylabel("PCA Feature #1")
ax.set_xlabel("PCA Feature #0")
ax.set_title("Training data")
plt.show()
################
## Kernel PCA 3D##
################
transformer = KernelPCA(n_components=3, 
                        kernel=kernell,
                        gamma=None) 
                        #, fit_inverse_transform=True, alpha=0.1)
X_transformed = transformer.fit_transform(data)
print(X_transformed.shape)
###############
## plot results
###############
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], s=s, color = color, zorder=2)
###################
# TSNE 2D ############
###################
tsne = TSNE(n_components=2, verbose=1, random_state=10)
z = tsne.fit_transform(data) 
X_transformed = z
###############
## plot results
###############
fig, ax = plt.subplots()
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], s=s, color = color, zorder=2)
ax.set_ylabel("TSNE Feature #1")
ax.set_xlabel("TSNE Feature #0")
ax.set_title("Training data")
plt.show()
###
###################
# TSNE 3D ############
###################
tsne = TSNE(n_components=3, verbose=1, random_state=10)
z = tsne.fit_transform(data) 
X_transformed = z
###############
## plot results
###############
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], s=s, color = color, zorder=2)
###################
# MSD 2D ############
###################
embedding = MDS(n_components=2, normalized_stress='auto')
X_transformed = embedding.fit_transform(data)
###############
## plot results
###############
fig, ax = plt.subplots()
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], s=s, color = color, zorder=2)
ax.set_ylabel("TSNE Feature #1")
ax.set_xlabel("TSNE Feature #0")
ax.set_title("Training data")
plt.show()
###
###################
# MSD 3D ############
###################
embedding = MDS(n_components=3, 
                # normalized_stress='auto'
                )
X_transformed = embedding.fit_transform(data)
###############
## plot results
###############
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], s=s, color = color, zorder=2)
###########################
# df = pd.DataFrame()
# df["y"] = y
# df["comp-1"] = z[:,0]
# df["comp-2"] = z[:,1]

# sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                 palette=sns.color_palette("hls", 3),
#                 data=df).set(title="Iris data T-SNE projection") 