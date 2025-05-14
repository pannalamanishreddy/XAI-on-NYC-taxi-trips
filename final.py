# NYC Taxi Fare Prediction with XAI Methods

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import joblib

import lime
import lime.lime_tabular
from treeinterpreter import treeinterpreter as ti
import shap
import warnings
warnings.filterwarnings('ignore')

# Load dataset
file_path = 'yellow_tripdata_2015-01.csv'
df = pd.read_csv(file_path, nrows=100000)

# Data Cleaning
df = df.dropna()
df = df[(df['fare_amount'] > 0) & (df['passenger_count'] > 0)]

# Feature Engineering
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df['distance_km'] = haversine_distance(df['pickup_latitude'], df['pickup_longitude'],
                                       df['dropoff_latitude'], df['dropoff_longitude'])

df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
df['hour'] = df['tpep_pickup_datetime'].dt.hour
df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek

features = ['passenger_count', 'distance_km', 'hour', 'day_of_week']
X = df[features]
y = df['fare_amount']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Save model
joblib.dump(model, 'nyc_taxi_fare_model.joblib')

# === 1. Random Forest Feature Importances ===
plt.figure(figsize=(8, 4))
plt.barh(features, model.feature_importances_)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.tight_layout()
plt.show()

# === 2. SHAP ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("SHAP Summary Plot:")
shap.summary_plot(shap_values, X_test, feature_names=features)

# Optional force plot (only works in Jupyter)
# shap.initjs()
# shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# === 3. Permutation Importance ===
perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = perm.importances_mean.argsort()

plt.figure(figsize=(8, 4))
plt.barh(np.array(features)[sorted_idx], perm.importances_mean[sorted_idx])
plt.xlabel('Permutation Importance')
plt.title('Permutation Feature Importance')
plt.tight_layout()
plt.show()

# === 4. Partial Dependence Plots (PDP) ===
PartialDependenceDisplay.from_estimator(model, X_test, features=['distance_km', 'hour'], kind="average")
plt.suptitle('Partial Dependence Plots (PDP)')
plt.tight_layout()
plt.show()

# === 5. ICE Plots ===
PartialDependenceDisplay.from_estimator(model, X_test, features=['distance_km'], kind="individual")
plt.suptitle('ICE Plot for Distance (Individual Conditional Expectation)')
plt.tight_layout()
plt.show()

# === 6. LIME (Local Explanation) ===
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=features,
    mode='regression',
    random_state=42
)

sample_idx = 0
sample = X_test.iloc[[sample_idx]]
sample_pred = model.predict(sample)

lime_exp = lime_explainer.explain_instance(
    data_row=sample.values[0],
    predict_fn=model.predict,
    num_features=len(features)
)

print(f"LIME Explanation for index {sample_idx}")
print(f"Actual fare: {y_test.iloc[sample_idx]:.2f}")
print(f"Predicted fare: {sample_pred[0]:.2f}")

lime_exp.as_pyplot_figure()
plt.title('LIME Explanation for Single Prediction')
plt.tight_layout()
plt.show()

# === 7. TreeInterpreter (Local Explanation for Tree Models) ===
prediction, bias, contributions = ti.predict(model, sample.values)

print(f"\nTreeInterpreter (index {sample_idx}):")
print(f"Bias (average prediction): {bias[0]:.2f}")
for fname, contrib in zip(features, contributions[0]):
    print(f"{fname}: contribution = {contrib:.2f}")

plt.figure(figsize=(8, 4))
plt.barh(features, contributions[0])
plt.xlabel('Contribution to Prediction')
plt.title('TreeInterpreter Feature Contributions')
plt.tight_layout()
plt.show()
