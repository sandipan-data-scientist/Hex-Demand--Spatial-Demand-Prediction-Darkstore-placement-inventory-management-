# Run this script once before starting the API or the dashboard.
# It trains the model and saves it to artifacts/model.pkl

import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- 1. Synthetic data (mirrors the notebook exactly) ----

np.random.seed(42)

center_lat, center_lon = 22.5726, 88.3639
n_orders = 5000

latitudes  = center_lat + np.random.normal(0, 0.02, n_orders)
longitudes = center_lon + np.random.normal(0, 0.02, n_orders)

start_time = datetime(2025, 1, 1)
timestamps = [
    start_time + timedelta(minutes=int(np.random.randint(0, 1440)))
    for _ in range(n_orders)
]
order_values = np.random.gamma(shape=2, scale=200, size=n_orders)

df = pd.DataFrame({
    'latitude'   : latitudes,
    'longitude'  : longitudes,
    'timestamp'  : timestamps,
    'order_value': order_values,
})

# ---- 2. Cleaning ----

df.drop_duplicates(inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour']      = df['timestamp'].dt.hour

# ---- 3. H3 indexing ----

import h3

resolution = 8
df['h3_index'] = df.apply(
    lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], resolution),
    axis=1
)

# ---- 4. Aggregate demand per hexagon per hour ----

hex_demand = (
    df.groupby(['h3_index', 'hour'])
    .size()
    .reset_index(name='demand')
)

# ---- 5. Lag feature ----

hex_demand.sort_values(by=['h3_index', 'hour'], inplace=True)
hex_demand['lag_1'] = hex_demand.groupby('h3_index')['demand'].shift(1)
hex_demand.dropna(inplace=True)
hex_demand.reset_index(drop=True, inplace=True)

# ---- 6. Train model ----

X = hex_demand[['lag_1', 'hour']]
y = hex_demand['demand']

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

# ---- 7. Quick evaluation printout ----

y_pred = model.predict(X)
print(f"Training MAE  : {mean_absolute_error(y, y_pred):.4f}")
print(f"Training RMSE : {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
print(f"Training R2   : {r2_score(y, y_pred):.4f}")

# ---- 8. Add predictions and save processed CSV ----

hex_demand['predicted_demand'] = y_pred

os.makedirs('artifacts', exist_ok=True)
hex_demand.to_csv('artifacts/hex_demand_processed.csv', index=False)
print("Saved: artifacts/hex_demand_processed.csv")

# ---- 9. Save model pickle using joblib ----
# joblib is preferred over pickle for scikit-learn models
# because it handles large numpy arrays more efficiently

joblib.dump(model, 'artifacts/model.pkl')
print("Saved: artifacts/model.pkl")