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
from scipy.stats import entropy
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
import shap

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

supercon_train = pd.read_csv("/data/angel/Superconductivity/data/unique_m.csv")
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

zhang_train = pd.read_csv("/data/angel/Superconductivity/data/12340_all.csv")
zhang_features = zhang_train.copy()
zhang_labels = zhang_features.pop('TC')
zhang_features.pop('DOPPED')
zhang_features = np.array(zhang_features)
zhang_labels = np.array(zhang_labels)

formula=pd.read_csv("/work/mroitegui/Superconductors/data/unique_m.csv")
# electrones=pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elements.csv")
electrones=pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elementswithelectronstotal.csv")
electronegativitycsv =pd.read_csv("/work/mroitegui/Superconductors/data/periodic_table_of_elements.csv")
superconductors_list=formula['material'].tolist()
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
entropy_atvec=np.ndarray(shape=(21263,1))
psd_atvec=np.ndarray(shape=(21263,1))
sumatvec=np.ndarray(shape=(21263,1))


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
electronegativity_list =np.array(electronegativity_list)
electronegativity_list=electronegativity_list.reshape(-1,1)
mendeleievlist = np.array(mendeleievlist)
mendeleievlist=mendeleievlist.reshape(-1,1)
masalist=np.array(masalist)
masalist=masalist.reshape(-1,1)
structure_list = np.array(structure_list)
structure_list = structure_list.reshape(-1,1)

for i in range(0,21263):
    # entropy_atvec[i,:] = entropy(supercon_features[i,:],base=2)
    entropy_atvec[i,:] = entropy(supercon_features[i,:])
    sumatvec[i,:] = np.sum(supercon_features[i,:])
    # freqs, psd = signal.welch(supercon_features[i,:])
    # psd_atvec[i,:] = np.max(psd)
    # lap_entropy_atvec = entropy(ndimage.laplace(supercon_features[i,:]),base=2)
    
maradius = magpie_features2.pop("mean_atomic_radius")
maradius = np.array(maradius)
maradius = maradius.reshape(-1,1)
meaf = magpie_features2.pop("mean_ElectronAffinity")
meaf = np.array(meaf)
meaf = meaf.reshape(-1,1)
mtc = magpie_features2.pop("mean_ThermalConductivity")
mtc = np.array(mtc)
mtc = mtc.reshape(-1,1)
rtc = magpie_features2.pop("range_ThermalConductivity")
rtc = np.array(rtc)
rtc = rtc.reshape(-1,1)
mval = magpie_features2.pop('mean_Valence')
mval = np.array(mval)
mval = mval.reshape(-1,1) 
matmas = magpie_features2.pop('mean_atomic_mass')
matmas = np.array(matmas)
matmas = matmas.reshape(-1,1)
mfie = magpie_features2.pop('mean_fie')
mfie = np.array(mfie)
mfie = mfie.reshape(-1,1) 
mdens = magpie_features2.pop('mean_Density')
mdens = np.array(mdens)
mdens = mdens.reshape(-1,1) 
mfh  = magpie_features2.pop('mean_FusionHeat')
mfh = np.array(mfh)
mfh = mfh.reshape(-1,1) 
####
adsradius = (np.pi*maradius**2)
adsradius = adsradius.reshape(-1,1)
adsvol = (4*np.pi*maradius**3)/3
adsvol = adsvol.reshape(-1,1)
gibbs = (maradius/2) - (np.pi*maradius**2)*(1/4*np.pi*maradius)
gibbs = gibbs.reshape(-1,1)
####
# TRAINING MATRIX #
#####

# X = supercon_features
# X = magpie_features
X = zhang_features
# X = np.concatenate((electronesslist, electronesplist, electronesdlist, electronesflist), axis= 1)
# X = np.concatenate((densidadslist, densidadplist, densidaddlist, densidadflist), axis= 1)
# X = np.concatenate((densidadslist, densidadplist, densidaddlist, densidadflist, 
# #                     entropy_atvec, electronegativity_list, masalist, mendeleievlist, structure_list), axis= 1) 
# X = np.concatenate((densidadslist, densidadplist, densidaddlist, densidadflist, #entropy_atvec, structure_list, masalist
#                      mval, electronegativity_list, mendeleievlist, maradius, masalist), axis= 1) 

X = np.concatenate((densidadslist, densidadplist, densidadflist, #entropy_atvec, structure_list, masalist
                    entropy_atvec, mval, electronegativity_list, mendeleievlist, maradius, masalist), axis= 1) 
####
y = zhang_labels
# y = supercon_labels
# y = magpie_labels
####
# from sklearn.preprocessing import Normalizer
# transformer = Normalizer().fit(X)
# transformer.transform(X)
# # transformer.transform(zhang_labels)




#######
#######
#NORMALIZATION
#######
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
# load data
data1 = X
data2 = y
data2 = data2.reshape(-1, 1)
# create scaler
scaler = MinMaxScaler()
# fit scaler on data
scaler.fit(data1)
Xn = scaler.transform(data1)
scaler.fit(data2)
yn = scaler.transform(data2)
######



# X_train, X_test, y_train, y_test = train_test_split(Xn[:,:20], yn, test_size=0.15, random_state=20)
X_train, X_test, y_train, y_test = train_test_split(Xn, yn, test_size=0.15, random_state=20)


################
## Kernel PCA ##
################
##  kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’}, default=’linear’
kernell = "rbf"
s = 0.1
color = 'b'
transformer = KernelPCA(n_components=2, 
                        kernel=kernell,
                        gamma=None) 
                        #, fit_inverse_transform=True, alpha=0.1)
X_transformed = transformer.fit_transform(X_train)
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

transformer = KernelPCA(n_components=3, 
                        kernel=kernell,
                        gamma=None) 
                        #, fit_inverse_transform=True, alpha=0.1)
X_transformed = transformer.fit_transform(X_train)
print(X_transformed.shape)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], s=s, color = color, zorder=2)




