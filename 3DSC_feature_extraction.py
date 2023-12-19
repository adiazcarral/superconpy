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
# cuprates=pd.read_csv('/Users/angel/work/superconpy/martin/cupratos.csv')
# ironbase=pd.read_csv('/Users/angel/work/superconpy/martin/ironbase.csv')
# lowTcuprate=pd.read_csv('/Users/angel/work/superconpy/martin/lowT.csv')
# highTcuprate=pd.read_csv('/Users/angel/work/superconpy/martin/highT.csv')
cuprates=pd.read_csv('martin/cupratos.csv')
ironbase=pd.read_csv('martin/ironbase.csv')
lowTcuprate=pd.read_csv('martin/lowT.csv')
highTcuprate=pd.read_csv('martin/highT.csv')
# folder_path='/work/mroitegui/Superconductivity/shap'
###############################################
# Define the CSV file path for 3DSC
csv_path = "/data/angel/Superconductivity/3DSC/superconductors_3D/data/final/MP/3DSC_MP.csv"
##############################################
# ## Formula for Zhang dataset ##
# formula = pd.read_csv("data/12340_all_pred.csv")
# #
# electrones=pd.read_csv("data/periodic_table_of_elementswithelectronstotal.csv")
# superconductors_list=formula['DOPPED'].tolist()
###############################
## Formula for 3DSC dataset ###
# Read the CSV file using pandas, skipping the first row for data and using the second row for column names
df_3DSC = pd.read_csv(csv_path, skiprows=[0]) 
formula = pd.read_csv(csv_path, skiprows=[0]) 
electrones=pd.read_csv("data/periodic_table_of_elementswithelectronstotal.csv")
superconductors_list=df_3DSC[('formula_sc')].tolist()
###############################

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
# Extracting lattice vectors, volume, and symmetry group
lattice_vector_a = df_3DSC[('lata_2')].values
lattice_vector_a = lattice_vector_a.reshape(-1,1)
lattice_vector_b = df_3DSC[('latb_2')].values
lattice_vector_b = lattice_vector_b.reshape(-1,1)
lattice_vector_c = df_3DSC[('latc_2')].values
lattice_vector_c = lattice_vector_c.reshape(-1,1)
density = df_3DSC[('density_2')].values
density = density.reshape(-1,1)
crystal_system = df_3DSC[('crystal_system_2')].values
crystal_system = crystal_system.reshape(-1,1)

# Convert lattice vectors to a single array
lattice_vectors = np.column_stack((lattice_vector_a, lattice_vector_b, lattice_vector_c))

# # Print or use individual numpy arrays
# print("Lattice Vector A:")
# print(lattice_vector_a)

# print("Lattice Vector B:")
# print(lattice_vector_b)

# print("Lattice Vector C:")
# print(lattice_vector_c)

# print("Volume:")
# print(density)

# print("Symmetry Group:")
# print(crystal_system)

# # Combine the extracted information into a numpy array
# superconductor_data = np.column_stack((lattice_vectors, density, crystal_system))

# # Print or use the 'superconductor_data' array as needed
# print("Combined Superconductor Data:")
# print(superconductor_data)
##########################################
#########################################
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
                            'Mend', 
                            'Mass',
                            'EN',
                            'Maradius',
                            'Mval',
                            'Mtc',
                            'Mfie',
                            'Mec',
                            'Smix',
                            'delta',
                        
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
                    Mend, 
                    Masa,
                    EN,
                    Maradius,
                    Mval,
                    Mtc,
                    Mfie,
                    Mec,
                    smix,
                    Delta,
                    ), axis= 1) 
### Labels Zhang ##
# y = zhang_labels
# print(np.shape(X))
### ------------ ##
#
## Labels 3DSC ##
y = df_3DSC[('tc')].values
yp = df_3DSC[('tc')].values
## ----------- ##
###
Xp = np.concatenate((
                    vecs[:,[
                        0,4,5,
                            8,
                            9,
                            10,
                            12,
                            13,14,15,16,17,18,19,20,21,22,23,24,25
                            ]],                  
                    Mend, 
                    Masa,
                    EN,
                    Maradius,
                    Mval,
                    Mtc,
                    Mfie,
                    Mec,
                    smix,
                    Delta,
                    lattice_vector_a,
                    lattice_vector_b,
                    lattice_vector_c,
                    density,
                    crystal_system,
                    y.reshape(-1,1)
                    ), axis= 1) 
###
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

###############
## plot results
###############
# df=pd.DataFrame(Xp[:,(3,30)])
# df['c']='O'
# # cuprates_list=[]
# # ironbase_list=[]
# colors={'HC':'tab:blue','LC':'tab:green' ,'I':'tab:orange', 'O':'tab:grey'}
# for i in highTcuprate['DOPPED']:
#     index=superconductors_list.index(i)
#     df.loc[index,'c']='HC'
    
# for i in lowTcuprate['DOPPED']:
#     index=superconductors_list.index(i)
#     df.loc[index,'c']='LC'
    
# for i in ironbase['DOPPED']:
#     index=index=superconductors_list.index(i)
#     df.loc[index,'c']='I'
# fig, ax = plt.subplots()
# ax.scatter(df[0], df[1],
#             s=0.5,
#             c=df['c'].map(colors),
#             zorder=2)
# ax.set_xlabel("$\Delta S_{mix}$", fontsize = 20)
# ax.set_ylabel("$T_c$", fontsize = 14)
# # ax.set_title("Training data")
# plt.show()
# #
# f, ax = plt.subplots()
# points = ax.scatter(smix, vecs[:,8], c=zhang_labels, s=0.1, cmap="plasma")
# # points = ax.scatter(smix, zhang_labels, c=zhang_labels, s=0.1, cmap="plasma")
# f.colorbar(points)
# plt.xlabel("$\Delta S_{mix}$", fontsize = 20)
# plt.ylabel("Mean Valence EC (Mval)", fontsize = 14)
# plt.show()
# #############################
#############################

f, ax = plt.subplots()
points = ax.scatter(density, Mec, c=yp, s=0.1, cmap="plasma")
# points = ax.scatter(smix, zhang_labels, c=zhang_labels, s=0.1, cmap="plasma")
f.colorbar(points)
# plt.xlabel("$\Delta S_{mix}$", fontsize = 20)
# plt.ylabel("Mean Valence EC (Mval)", fontsize = 14)
plt.show()