import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import featurefunctions as F
from vecs import compute_vecs
import tsfel
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D

# Load data
zhang_train = pd.read_csv("data/12340_all_pred.csv")
cuprates = pd.read_csv('/Users/angel/work/superconpy/martin/cupratos.csv')
ironbase = pd.read_csv('/Users/angel/work/superconpy/martin/ironbase.csv')
lowTcuprate = pd.read_csv('/Users/angel/work/superconpy/martin/lowT.csv')
highTcuprate = pd.read_csv('/Users/angel/work/superconpy/martin/highT.csv')
zhang_train = pd.read_csv("data/12340_all_pred.csv")
zhang_features = zhang_train.copy()
zhang_labels = zhang_features.pop('TC')
zhang_features.pop('DOPPED')
zhang_features = np.array(zhang_features)
zhang_labels = np.array(zhang_labels)
# Feature extraction (not shown)
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

# Cluster visualization (TSNE in this case)
tsne = TSNE(n_components=2, verbose=1, random_state=10)
z = tsne.fit_transform(data)

# Plotting
df = pd.DataFrame(z, columns=['TSNE Feature #0', 'TSNE Feature #1'])
df['Group'] = 'Others'

# Assigning groups based on superconductors
groups = {'High-T Cuprates': highTcuprate['DOPPED'], 'Low-T Cuprates': lowTcuprate['DOPPED'],
          'Iron-Based': ironbase['DOPPED']}

for group_name, group_data in groups.items():
    indices = [superconductors_list.index(i) for i in group_data if i in superconductors_list]
    df.loc[indices, 'Group'] = group_name

# Plotting
fig, ax = plt.subplots()
colors = {'High-T Cuprates': 'tab:blue', 'Low-T Cuprates': 'tab:green', 'Iron-Based': 'tab:orange', 'Others': 'tab:grey'}
for group, color in colors.items():
    ax.scatter(df.loc[df['Group'] == group, 'TSNE Feature #0'], df.loc[df['Group'] == group, 'TSNE Feature #1'],
               label=group, c=color, s=0.5)

# Creating custom legend markers
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='High-T Cuprates', markerfacecolor='tab:blue', markersize=7),
    Line2D([0], [0], marker='o', color='w', label='Low-T Cuprates', markerfacecolor='tab:green', markersize=7),
    Line2D([0], [0], marker='o', color='w', label='Iron-Based', markerfacecolor='tab:orange', markersize=7),
    Line2D([0], [0], marker='o', color='w', label='Others', markerfacecolor='tab:grey', markersize=7)
]

# Adding legend
ax.legend(handles=legend_elements, scatterpoints=1)


# Adding labels and title
ax.set_ylabel("t-SNE Feature 1", fontsize=15)
ax.set_xlabel("t-SNE Feature 0", fontsize=15)
# ax.set_title("Superconductors Clustering")

# Increasing the size of the tick labels
ax.tick_params(axis='both', which='major', labelsize=13)

# Save the figure with DPI 600
plt.savefig("superconductors_clustering.png", dpi=600, bbox_inches='tight')

# Show plot
plt.show()
