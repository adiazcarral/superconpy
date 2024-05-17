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
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import tsfel
from sklearn.decomposition import PCA, KernelPCA
import seaborn as sb
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
import umap.umap_ as umap
#####
zhang_train = pd.read_csv("data/12340_all_pred.csv")
zhang_features = zhang_train.copy()
zhang_labels = zhang_features.pop('TC')
zhang_features.pop('DOPPED')
zhang_features = np.array(zhang_features)
zhang_labels = np.array(zhang_labels)
cuprates=pd.read_csv('/work/mroitegui/Superconductors/cupratos.csv')
ironbase=pd.read_csv('/work/mroitegui/Superconductors/ironbase.csv')
lowTcuprate=pd.read_csv('/work/mroitegui/Superconductors/lowT.csv')
highTcuprate=pd.read_csv('/work/mroitegui/Superconductors/highT.csv')
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
    '1s','2s','2p',
           '3s',
            '3p',
            '3d',
            '4s',
            '4p','4d','4f','5s','5p','5d','5f',
            '6s',
            '6p','6d','6f',
            '7s',
            '7p',
                            # 'Mend', 
                            # 'Mass',
                            # 'EN',
                            # 'Maradius',
                            # 'Mval',
                            # 'Mtc',
                            # 'Mfie',
                            # 'Mec',
                            # 'Smix',
                            # 'delta',
                        
                        ]
X = np.concatenate((
                    vecs[:,[
                        0,4,5,
                            8,
                            9,
                            10,
                            12,
                            13,14,15,16,17,18,19,20,21,22,23,24,25
                            ]],                  
                    # Mend, 
                    # Masa,
                    # EN,
                    # Maradius,
                    # Mval,
                    # Mtc,
                    # Mfie,
                    # Mec,
                    # smix,
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
Xn = Xn[0:12340, :]
yn = yn[0:12340]
######
################
# CLUSTERING ###
################
data = Xn
###############
## plot results
###############
# fig = plt.figure()
# ax1 = fig.add_subplot(projection='3d')
# ax1.scatter(Xn[:, 0], Xn[:, 1], Xn[:, 2], s=0.05, color = 'b', zorder=2)
# plt.show()
#
# plt.scatter(X[:, 0], y,s=0.1, color = 'g', zorder=2)
# plt.show()
# # 
# plt.scatter(Xn[:, 0], Xn[:, 1],s=0.5, color = 'g', zorder=2)
# plt.show()
# plt.scatter(Xn[:, 0], Xn[:, 2],s=0.05, color = 'b', zorder=2)
# plt.show()
# plt.scatter(Xn[:, 2], Xn[:, 1],s=0.05, color = 'b', zorder=2)
# plt.show()
# ################
# ## Kernel PCA 2D##
# ################
# ##  kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’}, default=’linear’
# kernell = "sigmoid"
# s = 0.05
# color = 'b'
# transformer = KernelPCA(n_components=2, 
#                         kernel=kernell,
#                         gamma=None) 
#                         #, fit_inverse_transform=True, alpha=0.1)
# # X_transformed = transformer.fit_transform(X_train)
# X_transformed = transformer.fit_transform(data)
# print(X_transformed.shape)

# ###############
# ## plot results
# ###############
# fig, ax = plt.subplots()
# ax.scatter(X_transformed[:, 0], X_transformed[:, 1], s=s, color = color, zorder=2)
# ax.set_ylabel("PCA Feature #1")
# ax.set_xlabel("PCA Feature #0")
# ax.set_title("Training data")
# plt.show()
# # ################
# # ## Kernel PCA 3D##
# # ################
# # transformer = KernelPCA(n_components=3, 
# #                         kernel=kernell,
# #                         gamma=None) 
# #                         #, fit_inverse_transform=True, alpha=0.1)
# # X_transformed = transformer.fit_transform(data)
# # print(X_transformed.shape)
# # ###############
# # ## plot results
# # ###############
# # fig = plt.figure()
# # ax = fig.add_subplot(projection='3d')
# # ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], s=s, color = color, zorder=2)
# ###################

# # TSNE 2D ############
# ###################
# Initialize t-SNE object with custom parameters
tsne = TSNE(n_components=2, 
            perplexity=70, 
            learning_rate=200, 
            n_iter=1000,
            verbose = 1,
            random_state=42)
z = tsne.fit_transform(data) 
X_transformed = z
###############
## plot results
###############
df=pd.DataFrame(X_transformed)
df['c']='O'
# cuprates_list=[]
# ironbase_list=[]
colors={'HC':'tab:blue','LC':'tab:green' ,'I':'tab:orange', 'O':'tab:grey'}
for i in highTcuprate['DOPPED']:
    index=superconductors_list.index(i)
    df.loc[index,'c']='HC'
    
for i in lowTcuprate['DOPPED']:
    index=superconductors_list.index(i)
    df.loc[index,'c']='LC'
    
for i in ironbase['DOPPED']:
    index=index=superconductors_list.index(i)
    df.loc[index,'c']='I'
fig, ax = plt.subplots()
ax.scatter(df[0], df[1],
            s=0.5,
           c=df['c'].map(colors),
           zorder=2)
ax.set_ylabel("TSNE Feature #1")
ax.set_xlabel("TSNE Feature #0")
# ax.set_title("Training data")
plt.show()
# fig, ax = plt.subplots()
# ax.scatter(X_transformed[:, 0], X_transformed[:, 1], s=s, color = color, zorder=2)
# ax.set_ylabel("TSNE Feature #1")
# ax.set_xlabel("TSNE Feature #0")
# ax.set_title("Training data")
# plt.show()
# ###
# # ###################
# # # TSNE 3D ############
# # ###################
# # tsne = TSNE(n_components=3, verbose=1, random_state=10)
# # z = tsne.fit_transform(data) 
# # X_transformed = z
# # ###############
# # ## plot results
# # ###############
# # fig = plt.figure()
# # ax = fig.add_subplot(projection='3d')
# # ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], s=s, color = color, zorder=2)
# ###############
# ### UMAP -----#
# ------------#
# clusterable_embedding = umap.UMAP(
#     n_neighbors=30,
#     min_dist=0.0,
#     n_components=2,
#     random_state=42,
# ).fit_transform(data)
# X_transformed = clusterable_embedding
###############
## plot results
###############

# fig, ax = plt.subplots()
# ax.scatter(df[0], df[1],
#            # s=s,
#            c=df['c'].map(colors),
#            zorder=2)
# ax.set_ylabel("UMAP Feature #1")
# ax.set_xlabel("UMAP Feature #0")
# ax.set_title("Training data")
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
#            # s=s, color = color,
#            zorder=2)
# ax.set_ylabel("UMAP Feature #1")
# ax.set_xlabel("UMAP Feature #0")
# ax.set_title("Training data")
# plt.show()
###########################
# df = pd.DataFrame()
# df["y"] = y
# df["comp-1"] = z[:,0]
# df["comp-2"] = z[:,1]

# sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                 palette=sns.color_palette("hls", 3),
#                 data=df).set(title="Iris data T-SNE projection") 