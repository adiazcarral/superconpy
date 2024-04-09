import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tsfel
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import umap.umap_ as umap
from vecs import compute_vecs
import featurefunctions as F
# Load data
zhang_train = pd.read_csv("data/12340_all_pred.csv")
zhang_features = zhang_train.copy()
zhang_labels = zhang_features.pop('TC')
zhang_features.pop('DOPPED')
zhang_features = np.array(zhang_features)
zhang_labels = np.array(zhang_labels)
cuprates=pd.read_csv('martin/cupratos.csv')
ironbase=pd.read_csv('martin/ironbase.csv')
lowTcuprate=pd.read_csv('martin/lowT.csv')
highTcuprate=pd.read_csv('martin/highT.csv')
# folder_path='/work/mroitegui/Superconductivity/shap'
##########
formula = pd.read_csv("data/12340_all_pred.csv")
electrones=pd.read_csv("data/periodic_table_of_elementswithelectronstotal.csv")
superconductors_list=formula['DOPPED'].tolist()

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

# Remove outliers
outliers_smix = np.argwhere(np.isnan(smix))
y = zhang_labels
Xp = np.concatenate((
    vecs[:, [
        0, 4, 5,
        8,
        9,
        10,
        12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
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
    y.reshape(-1, 1)
), axis=1)
X = Xp[:, :-1]  # Remove last column (Tc)

# Feature selection
X = pd.DataFrame(X)
corr_features = tsfel.correlated_features(X)
X.drop(corr_features, axis=1, inplace=True)
selector = VarianceThreshold()
X = selector.fit_transform(X)

# Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Plotting
df = pd.DataFrame(Xp[:, (2, 3, 5, 8, 24, 30)])
df['c'] = 'O'

colors = {'HC': 'tab:blue', 'LC': 'tab:green', 'I': 'tab:orange', 'O': 'tab:grey'}
for i in highTcuprate['DOPPED']:
    index = superconductors_list.index(i)
    df.loc[index, 'c'] = 'HC'

for i in lowTcuprate['DOPPED']:
    index = superconductors_list.index(i)
    df.loc[index, 'c'] = 'LC'

for i in ironbase['DOPPED']:
    index = index = superconductors_list.index(i)
    df.loc[index, 'c'] = 'I'

# Plot Tc vs Orbitals
fig, axs = plt.subplots(2, 2, figsize=(12, 9))

orbital_positions = [4, 1, 0, 2]  # Positions of orbitals in Xp
orbital_labels = ['Mval', '3s', '2p', '3d']  # Labels for orbitals
for i, feature_pos in enumerate(orbital_positions):
    axs[i // 2, i % 2].scatter(df.iloc[:, feature_pos], y, c=df['c'].map(colors), s=5)
    axs[i // 2, i % 2].set_xlabel(f"{orbital_labels[i]}", fontsize=16)
    axs[i // 2, i % 2].set_ylabel("$T_c$", fontsize=16)
    axs[i // 2, i % 2].set_title(f"{orbital_labels[i]} vs Tc", fontsize=16)
    
# Add legend
for group, color in colors.items():
    axs[1, 1].scatter([], [], c=color, label=group)
axs[1, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)

plt.tight_layout()
plt.savefig("Tc_vs_Orbitals_Mval.png", dpi=600)
plt.show()

# Plot smix vs Orbitals without Tc colormap
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for i, feature_pos in enumerate(orbital_positions):
    sc = axs[i // 2, i % 2].scatter(smix, df.iloc[:, feature_pos], c=y, cmap='plasma', s=5)
    axs[i // 2, i % 2].scatter(smix, df.iloc[:, feature_pos], c=y, cmap='plasma', s=5)
    axs[i // 2, i % 2].set_xlabel(f"{orbital_labels[i]}", fontsize=16)
    axs[i // 2, i % 2].set_ylabel("smix", fontsize=16)
    axs[i // 2, i % 2].set_title(f"smix vs {orbital_labels[i]}", fontsize=16)

# Add colorbar
cbar_ax = fig.add_axes([1.0, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = plt.colorbar(sc, cax=cbar_ax)
cbar.set_label('Tc', fontsize=16)

plt.tight_layout()
plt.savefig("smix_vs_Orbitals_Mval.png", dpi=600, bbox_inches='tight')  # Add bbox_inches='tight'
plt.show()

# Plot smix vs Orbitals with group colormap
fig, axs = plt.subplots(2, 2, figsize=(12, 9))

for i, feature_pos in enumerate(orbital_positions):
    sc = axs[i // 2, i % 2].scatter(smix, df.iloc[:, feature_pos], c=df['c'].map(colors), s=5)
    axs[i // 2, i % 2].set_xlabel(f"{orbital_labels[i]}", fontsize=16)
    axs[i // 2, i % 2].set_ylabel("smix", fontsize=16)
    axs[i // 2, i % 2].set_title(f"smix vs {orbital_labels[i]}", fontsize=16)

# Add legend
for group, color in colors.items():
    axs[1, 1].scatter([], [], c=color, label=group)
axs[1, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)


plt.tight_layout()
plt.savefig("smix_vs_Orbitals_Mval_group_colormap.png", dpi=600)
plt.show()