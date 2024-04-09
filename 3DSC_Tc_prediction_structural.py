import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import xgboost
import sklearn.metrics
import math
from matplotlib import pyplot as plt
import tsfel
# Import custom modules
from vecs import compute_vecs
import featurefunctions as F
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# Read the CSV file using pandas, skipping the first row for data and using the second row for column names
csv_path = "/data/angel/Superconductivity/3DSC/superconductors_3D/data/final/MP/3DSC_MP.csv"
df_3DSC = pd.read_csv(csv_path, skiprows=[0])
electrones = pd.read_csv("data/periodic_table_of_elementswithelectronstotal.csv")
superconductors_list = df_3DSC['formula_sc'].tolist()
formula = pd.read_csv(csv_path, skiprows=[0])
# Extracting purely numerical columns from the 3DSC CSV
numerical_columns = df_3DSC.select_dtypes(include=np.number).drop('tc', axis=1).values

# Labels
y = df_3DSC['tc'].values

# Calculate the average number of itinerant electrons per atom (e/a)
electronegativity_list = F.EN_old(formula, electrones, superconductors_list)
average_e_per_a = np.mean(electronegativity_list, axis=1)

# Add the new feature to the numerical columns
numerical_columns = np.column_stack((numerical_columns, average_e_per_a))

# FEATURE SELECTION
numerical_columns = pd.DataFrame(numerical_columns)
corr_features = tsfel.correlated_features(numerical_columns)
numerical_columns.drop(corr_features, axis=1, inplace=True)

# Remove low variance features
selector = VarianceThreshold()
numerical_columns = selector.fit_transform(numerical_columns)

# Convert numerical_columns back to DataFrame after feature selection
numerical_columns = pd.DataFrame(numerical_columns)

# NORMALIZATION
scaler = MinMaxScaler(feature_range=(0, 1))
Xn = scaler.fit_transform(numerical_columns)
yn = scaler.fit_transform(y.reshape(-1, 1))

# EXTRACT REAL Y_TEST
Xn_test = Xn[0:5772, :]
yn_test = yn[0:5772]

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(Xn_test, yn_test, test_size=0.15, random_state=15, shuffle=True)

# XGBoost
regr = xgboost.XGBRegressor(
    n_estimators=1000,  # 800,
    max_depth=16,  # 7
    eta=0.02,  # 0.1
    subsample=1,  # 0.7
    colsample_bytree=0.5,  # 0.8
    # booster='dart'
).fit(X_train, y_train)

# Feature importance
feature_importance = regr.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

# Select the two most relevant features
top_features = numerical_columns.columns[sorted_idx][:2]

# PREDICTION
y_pred = regr.predict(X_test)

# Replace negative predictions with a small positive value
small_value = 1e-10
y_pred[y_pred < 0] = small_value

y_pred = y_pred.reshape(-1, 1)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Plotting the two most relevant features
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, sorted_idx[0]], X_test[:, sorted_idx[1]], c=y_test, cmap='viridis', edgecolors='k', s=50)
plt.xlabel(top_features[0])
plt.ylabel(top_features[1])
plt.title('Two Most Relevant Features based on Feature Importance')
plt.colorbar(label='Measured TC')
plt.show()

# Plotting average number of itinerant electrons per atom (e/a) against Tc
plt.figure(figsize=(10, 6))
plt.scatter(average_e_per_a, y, c=y, cmap='viridis', edgecolors='k', s=50)
plt.xlabel('Average e/a')
plt.ylabel('Tc')
plt.title('Average e/a versus Tc')
plt.colorbar(label='Measured TC')
plt.show()

# Plotting average number of itinerant electrons per atom (e/a) against Tc
plt.figure(figsize=(10, 6))
plt.scatter(electronegativity_list, y, c=y, cmap='viridis', edgecolors='k', s=50)
plt.xlabel('Electronegativity')
plt.ylabel('Tc')
plt.title('Electronegativity versus Tc')
plt.colorbar(label='Measured TC')
plt.show()

# Additional evaluation and plotting code
print("\n")
print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test, y_pred))
print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test, y_pred))
print("Mean squared logarithmic error (MSLE): %f" % sklearn.metrics.mean_squared_log_error(y_test, y_pred))
print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred)))
print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test, y_pred))

MAE = np.absolute(y_test - y_pred)
MSE = np.square(y_test - y_pred)
RMSE = np.sqrt(MSE)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, s=0.4, color='k', zorder=2)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=3, zorder=1)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')

plt.show()
