# Hex-Demand: Predicting Grocery Demand Using H3 Spatial Indexing

A machine learning project that predicts grocery delivery demand density across
hexagonal city zones, inspired by how quick-commerce platforms like Zepto and
Blinkit manage dark store operations. Built end-to-end in Python: from synthetic
data generation to an interactive Folium map, a FastAPI prediction service, a
Streamlit dashboard, and full Docker containerisation.

---

## Table of Contents

1. [Business Context](#1-business-context)
2. [Objective](#2-objective)
3. [Tools and Libraries](#3-tools-and-libraries)
4. [Statistical Perspective](#4-statistical-perspective)
5. [Project Structure](#5-project-structure)
6. [Setup and Installation](#6-setup-and-installation)
7. [Notebook Walkthrough](#7-notebook-walkthrough)
   - [Importing Required Libraries](#71-importing-required-libraries)
   - [Creating a Synthetic Dataset](#72-creating-a-synthetic-dataset)
   - [Data Cleaning and Preprocessing](#73-data-cleaning-and-preprocessing)
   - [Converting Locations to H3 Hexagons](#74-converting-locations-to-h3-hexagons)
   - [Aggregating Demand per Hexagon](#75-aggregating-demand-per-hexagon)
   - [Feature Engineering: Lag Feature](#76-feature-engineering-lag-feature)
   - [Exploratory Data Analysis](#77-exploratory-data-analysis)
   - [Preparing Data for Machine Learning](#78-preparing-data-for-machine-learning)
   - [Model Selection and Training](#79-model-selection-and-training)
   - [Model Evaluation](#710-model-evaluation)
   - [Predicting Demand for Visualization](#711-predicting-demand-for-visualization)
   - [Converting H3 Indices to Geographic Coordinates](#712-converting-h3-indices-to-geographic-coordinates)
   - [Interactive Map Visualization Using Folium](#713-interactive-map-visualization-using-folium)
   - [Saving the Processed Dataset](#714-saving-the-processed-dataset)
8. [API Reference](#8-api-reference)
9. [Streamlit Dashboard](#9-streamlit-dashboard)
10. [Docker Deployment](#10-docker-deployment)
11. [Generated Artifacts](#11-generated-artifacts)
12. [Business Insights and Conclusion](#12-business-insights-and-conclusion)
13. [Future Enhancements](#13-future-enhancements)

---

## 1. Business Context

Last-minute grocery delivery platforms such as Zepto rely on accurate demand
forecasting to ensure fast deliveries. A 10-minute delivery promise is only
possible if the right inventory is at the right dark store before the customer
places the order.

To make this work at city scale, the city is divided into hexagonal grid cells
using Uber's H3 indexing system. Hexagons are preferred over squares or circles
because the distance from the center of any hexagon to the center of each of
its 6 neighbours is exactly equal. This geometric property eliminates distance
distortion and makes radius-based dark store coverage calculations precise.

This project builds a demand prediction system on top of that hexagonal grid.
By predicting how many orders each zone will receive in the next hour, operations
teams can pre-position riders and inventory before the demand actually arrives.

---

## 2. Objective

The goal of this project is to:

1. Data Preprocessing delivery location data (latitude and longitude) centered
   around Kolkata.
2. Convert these locations into H3 hexagonal indices at resolution 8.
3. Aggregate demand within each hexagon per hour.
4. Engineer a lag feature to capture temporal demand patterns.
5. Train a machine learning model to predict demand density per hexagon.
6. Visualize predicted high-demand areas using an interactive Folium map.
7. Serve predictions through a FastAPI REST API.
8. Display results on an interactive Streamlit dashboard.
9. Package the entire service in Docker for deployment.

---

## 3. Tools and Libraries

| Library | Version | Purpose |
|---|---|---|
| numpy | 1.26.4 | Numerical computations and random data generation |
| pandas | 2.2.2 | Data manipulation, aggregation, and feature engineering |
| h3 | 3.7.6 | H3 spatial indexing: lat/lon to hexagon conversion |
| scikit-learn | 1.4.2 | Random Forest model, train-test split, evaluation metrics |
| folium | 0.16.0 | Interactive Leaflet.js map generation |
| matplotlib | 3.8.4 | Static EDA plots |
| seaborn | 0.13.2 | Statistical EDA visualizations |
| fastapi | 0.111.0 | REST API for serving model predictions |
| uvicorn | 0.29.0 | ASGI server to run the FastAPI app |
| pydantic | 2.7.1 | Request and response schema validation for the API |
| streamlit | 1.35.0 | Interactive web dashboard |
| plotly | 5.22.0 | Interactive charts inside the Streamlit dashboard |
| joblib | 1.4.2 | Saving and loading the trained model pickle |

---

## 4. Statistical Perspective

Demand prediction is treated as a regression problem. The target variable is
the number of orders placed within a hexagon during a specific hour, referred
to as demand density.

The model minimises Mean Squared Error (MSE) during training:

```
MSE = (1/n) x sum( (y_actual - y_predicted)^2 )
```

Where n is the number of hex-hour records, y_actual is the observed order count,
and y_predicted is the model's output. MSE penalises large errors more than small
ones because of the squaring, which is appropriate here since severely
underpredicting demand in a high-order zone has a bigger operational cost than
a minor miss in a quiet zone.

The lag feature introduces an autoregressive component:

```
lag_1[t] = demand[t-1]
```

This reflects the real-world observation that if a zone was busy in the last
hour, it is more likely to be busy in the next hour. This is formally called
autocorrelation in time series analysis.

---

## 5. Project Structure

```
hex_demand/
│
├── Project_Hex_demand_forecast.ipynb   # the main notebook (full pipeline)
│
├── model/
│   └── train_and_save.py               # trains the model and saves artifacts
│
├── app/
│   ├── main.py                         # FastAPI routes
│   ├── predict.py                      # model loading and prediction logic
│   └── schemas.py                      # Pydantic request/response models
│
├── streamlit_app/
│   └── dashboard.py                    # Streamlit dashboard with all charts
│
├── artifacts/
│   ├── model.pkl                       # saved trained model (generated)
│   └── hex_demand_processed.csv        # processed hex-hour dataset (generated)
│
├── Dockerfile                          # containerises the FastAPI service
├── Dockerfile.streamlit                # containerises the Streamlit dashboard
├── docker-compose.yml                  # runs both services together
├── requirements.txt                    # all Python dependencies with versions
└── README.md                           # this file
```

The `artifacts/` folder is generated at runtime. It does not exist when you
first clone the repository. Running `model/train_and_save.py` creates it along
with both files inside it.

---

## 6. Setup and Installation

**Prerequisites:** Python 3.11 or higher, pip, and Docker (for containerised
deployment only).

**Clone or download the project:**

```bash
git clone https://github.com/your-username/hex-demand.git
cd hex_demand
```

**Create and activate a virtual environment:**

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS and Linux
source venv/bin/activate
```

**Install all dependencies:**

```bash
pip install -r requirements.txt
```

**Generate the model and processed data:**

```bash
python model/train_and_save.py
```

This runs the full notebook pipeline, trains the Random Forest model, and
saves two files into the `artifacts/` folder: `model.pkl` and
`hex_demand_processed.csv`. You must run this before starting the API or
the dashboard.

**Run the notebook:**

```bash
jupyter notebook Project_Hex_demand_forecast.ipynb
```

---

## 7. Notebook Walkthrough

The notebook `Project_Hex_demand_forecast.ipynb` is the core of this project.
Every section below mirrors a section inside the notebook in the same order.

### 7.1 Importing Required Libraries

The notebook begins by importing all required libraries in a single cell so
that any reader immediately knows the full dependency set.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import h3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import folium
from folium.plugins import HeatMap

np.random.seed(42)
```

The random seed is fixed at 42 throughout the project. This ensures that every
time the notebook is run, the synthetic data and model results are identical,
which is important for reproducibility during code review or interviews.

---

### 7.2 Creating a Synthetic Dataset

Since real delivery data is confidential, order locations are simulated within
Kolkata. The central coordinates of Kolkata (latitude 22.5726, longitude 88.3639)
are used as the anchor point.

**What the dataset contains:**

| Column | Description |
|---|---|
| latitude | Randomly generated latitude around Kolkata center |
| longitude | Randomly generated longitude around Kolkata center |
| timestamp | Random minute within a 24-hour window on 2025-01-01 |
| order_value | Synthetic order value drawn from a Gamma distribution |

**How the data is generated:**

Coordinates are drawn from a Normal distribution centred at Kolkata's city
center with a standard deviation of 0.02 degrees (approximately 2 kilometres).
This creates a realistic city-shaped cluster rather than a perfectly uniform
spread.

Order values are drawn from a Gamma distribution (shape=2, scale=200) because
order values in grocery delivery are right-skewed: most orders are small but
some are significantly larger. The Gamma distribution captures this shape
naturally.

```python
center_lat, center_lon = 22.5726, 88.3639
n_orders = 5000

latitudes  = center_lat + np.random.normal(0, 0.02, n_orders)
longitudes = center_lon + np.random.normal(0, 0.02, n_orders)

start_time = datetime(2025, 1, 1)
timestamps = [
    start_time + timedelta(minutes=np.random.randint(0, 1440))
    for _ in range(n_orders)
]
order_values = np.random.gamma(shape=2, scale=200, size=n_orders)

df = pd.DataFrame({
    'latitude'   : latitudes,
    'longitude'  : longitudes,
    'timestamp'  : timestamps,
    'order_value': order_values
})
```

Total records generated: 5000 synthetic orders across a 24-hour period.

---

### 7.3 Data Cleaning and Preprocessing

Data cleaning ensures the reliability of all downstream analysis. Even with
synthetic data, practising a proper cleaning routine is essential because the
same steps would be applied to real production data.

**Checks performed:**

- Missing values: `df.isnull().sum()` confirms no nulls are present.
- Duplicate records: `drop_duplicates()` removes any exact duplicate rows.
- Data types: the timestamp column is explicitly cast to `datetime64` using
  `pd.to_datetime()` to enable time-based operations.

**Feature extracted at this stage:**

The hour of day is extracted from the timestamp:

```python
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour']      = df['timestamp'].dt.hour
```

This `hour` column (integer 0 to 23) becomes one of the two features used in
the machine learning model. It captures the time-of-day demand cycle, for
example the difference between 3am (low demand) and 8pm (peak demand).

---

### 7.4 Converting Locations to H3 Hexagons

Each (latitude, longitude) pair is converted to an H3 hexagonal cell index
using `h3.latlng_to_cell()` at resolution 8.

**Why resolution 8:**

H3 resolution 8 produces hexagons that each cover approximately 0.74 square
kilometres. This is the right granularity for neighbourhood-level demand
analysis. Too coarse (resolution 6 or 7) and nearby but distinct demand zones
get merged together. Too fine (resolution 10 or 11) and most hexagons have
too few orders to compute meaningful statistics.

```python
resolution = 8

df['h3_index'] = df.apply(
    lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], resolution),
    axis=1
)
```

The resulting `h3_index` is a 15-character hexadecimal string that uniquely
identifies one hexagonal cell on the globe at resolution 8. All orders that
fall within the same hexagon share the same `h3_index` value. This string is
the spatial key used for all subsequent grouping operations.

**H3 version note:** `h3.latlng_to_cell()` is the API for h3-py version 4.x.
If you are running version 3.x, replace it with `h3.geo_to_h3(lat, lon, resolution)`.

---

### 7.5 Aggregating Demand per Hexagon

Individual order records are aggregated into (hexagon, hour) groups. Each row
in the resulting dataset represents one hexagon during one hour and contains
the count of orders that occurred in that zone during that hour.

This count is the target variable: demand.

```python
hex_demand = (
    df.groupby(['h3_index', 'hour'])
    .size()
    .reset_index(name='demand')
)
```

**Why this aggregation step matters:**

The raw dataset has one row per order (5000 rows). The aggregated dataset has
one row per (hexagon, hour) combination. Only the aggregated form is useful for
prediction because the goal is to predict how many orders a zone will receive
in the next hour, not to predict anything about individual orders. This
transformation is the bridge between raw event data and a forecastable signal.

---

### 7.6 Feature Engineering: Lag Feature

A lag feature captures temporal continuity in demand. The lag-1 feature for a
given hexagon at hour t is simply the demand that same hexagon recorded at
hour t-1.

```
lag_1[t] = demand[t-1]
```

```python
hex_demand.sort_values(by=['h3_index', 'hour'], inplace=True)
hex_demand['lag_1'] = hex_demand.groupby('h3_index')['demand'].shift(1)
hex_demand.dropna(inplace=True)
```

**Why this feature is the most important one in the model:**

Demand exhibits autocorrelation: a zone that was busy in the last hour is more
likely to stay busy in the next hour than a zone that had zero orders. The lag
feature gives the model a direct numerical signal of recent activity. Without
it, the model would only have the hour of day to work with, which is a much
weaker signal.

The `shift(1)` is applied within each hexagon group separately using
`groupby('h3_index')`. This prevents the demand from one hexagon at hour 5 from
being used as the lag feature for a completely different hexagon at hour 6,
which would be meaningless.

Rows where lag_1 is NaN (the very first hour recorded for each hexagon) are
dropped because there is no previous value available for them.

---

### 7.7 Exploratory Data Analysis

Two visualisations are produced to understand the data before modelling.

**Plot 1: Distribution of demand per hexagon**

A histogram of the `demand` column. This is expected to be right-skewed,
meaning most hexagons have low order counts per hour but a small number of
hexagons have very high counts. This is the Pareto pattern typical in urban
demand data.

```python
sns.histplot(hex_demand['demand'], bins=30)
plt.title('Distribution of Demand per Hexagon')
```

**Plot 2: Lag feature vs current demand scatter plot**

A scatter plot of lag_1 on the x-axis against demand on the y-axis. A positive
correlation here confirms that the lag feature carries genuine predictive
signal. If the scatter showed no pattern (a random cloud), the lag feature
would not be useful.

```python
sns.scatterplot(x='lag_1', y='demand', data=hex_demand)
plt.title('Lag Feature vs Current Demand')
```

A clear upward trend in this plot is what justified including lag_1 as a model
feature rather than discarding it.

---

### 7.8 Preparing Data for Machine Learning

The feature matrix X and target vector y are defined, and the dataset is split
into training and test sets using an 80-20 ratio.

```python
X = hex_demand[['lag_1', 'hour']]
y = hex_demand['demand']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Features used:**

| Feature | Type | What it captures |
|---|---|---|
| lag_1 | Float | Demand in the previous hour for this hexagon |
| hour | Integer (0-23) | Time of day demand cycle |

**Why only two features:**

The notebook keeps the feature set minimal and interpretable. Both features
have a clear, justifiable relationship to the target variable. Adding more
features (for example, order value statistics) without a clear causal reasoning
risks adding noise rather than signal at this data scale.

**Note on the split approach:**

The notebook uses a random 80-20 split via `train_test_split`. For a production
system handling real timestamped data, a temporal split (train on past dates,
test on future dates) would be more appropriate to avoid data leakage. The
`model/train_and_save.py` training script also uses a temporal split for this
reason.

---

### 7.9 Model Selection and Training

A Random Forest Regressor is used with 100 trees and a maximum tree depth of 10.

```python
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
```

**Why Random Forest:**

1. It captures non-linear relationships. The relationship between hour of day
   and demand is not linear: demand at hour 20 (8pm) is much higher than at
   hour 2 (2am), but the pattern does not follow a straight line.

2. It is robust to outliers. A few hexagons with unusually high demand counts
   will not distort the model the way they would a linear regression.

3. It reduces overfitting through ensemble averaging. Each tree is trained on
   a random subset of data and features. The final prediction is the average
   across all 100 trees, which smooths out the individual errors of any single
   tree.

**Mathematical formula:**

```
Prediction = (1/N) x sum of all tree predictions
```

Where N is the number of trees (100 in this project). The model internally
minimises MSE at each split point to decide which feature and threshold to
split on.

---

### 7.10 Model Evaluation

Three metrics are computed on the held-out test set:

```python
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
```

**What each metric means in plain terms:**

**MAE (Mean Absolute Error):**
The average absolute gap between what the model predicted and what actually
happened. If MAE is 1.8, the model's predictions are off by about 1.8 orders
on average. This is easy to explain to a non-technical stakeholder.

```
MAE = (1/n) x sum( |y_actual - y_predicted| )
```

**RMSE (Root Mean Squared Error):**
Similar to MAE but errors are squared before averaging, then square-rooted.
This means RMSE punishes large prediction errors more severely than MAE. If a
model predicted 5 orders but 20 arrived, that large miss contributes much more
to RMSE than to MAE. For operations planning, large misses are more costly than
small misses, so RMSE is the primary metric here.

```
RMSE = sqrt( (1/n) x sum( (y_actual - y_predicted)^2 ) )
```

**R-squared (R2 Score):**
The proportion of variance in demand that the model explains. An R2 of 0.85
means the model accounts for 85% of why demand varies across hexagons and
hours. The remaining 15% is variance the model cannot explain with just these
two features. R2 ranges from 0 (no explanatory power) to 1 (perfect fit).

```
R2 = 1 - (sum of squared residuals / total sum of squares)
```

---

### 7.11 Predicting Demand for Visualization

After evaluation on the test set, predictions are generated for the entire
hex_demand dataset so that every hexagon has a predicted_demand value available
for the map.

```python
hex_demand['predicted_demand'] = model.predict(X)
```

Note that `X` here is the full feature matrix, not just the test split. This
is intentional: the map shows a snapshot of predicted demand across all
hexagons, not just the 20% in the test set.

---

### 7.12 Converting H3 Indices to Geographic Coordinates

To draw hexagons on a Folium map, the H3 index must be converted back into a
list of geographic boundary coordinates (the six vertices of the hexagon).

```python
def h3_to_polygon(h3_index):
    boundary = h3.cell_to_boundary(h3_index)
    return [(lat, lon) for lat, lon in boundary]

hex_demand['polygon'] = hex_demand['h3_index'].apply(h3_to_polygon)
```

`h3.cell_to_boundary()` returns a list of six (latitude, longitude) tuples
representing the corners of the hexagon. Folium's `Polygon` object accepts this
list directly to draw the shape on the map.

---

### 7.13 Interactive Map Visualization Using Folium

An interactive map is created centred on Kolkata. Each hexagon is drawn as a
coloured polygon where the fill opacity is proportional to its predicted demand:
hexagons with higher predicted demand appear more intensely red, and those with
lower demand are nearly transparent.

```python
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

max_demand = hex_demand['predicted_demand'].max()

for _, row in hex_demand.iterrows():
    folium.Polygon(
        locations    = row['polygon'],
        color        = None,
        fill         = True,
        fill_color   = 'red',
        fill_opacity = row['predicted_demand'] / max_demand,
        popup        = f"Predicted Demand: {row['predicted_demand']:.2f}"
    ).add_to(m)

m.save('hex_demand_map.html')
```

The map is saved as `hex_demand_map.html`. This is a fully self-contained HTML
file that can be opened in any browser, shared by email, or embedded in a web
application without any Python running. Clicking on any hexagon shows a popup
with its predicted demand value.

**Why this output matters for the business:**

An operations manager looking at a spreadsheet of 300 hexagon IDs and demand
numbers cannot easily decide where to send riders. Looking at a colour-coded
city map, the high-demand zones are immediately obvious. This is the difference
between data and a decision.

---

### 7.14 Saving the Processed Dataset

The final aggregated and feature-engineered dataset is saved as a CSV for
reproducibility and for use by the API and dashboard.

```python
hex_demand.to_csv('hex_demand_processed.csv', index=False)
```

**Columns in the saved CSV:**

| Column | Description |
|---|---|
| h3_index | H3 hexagon identifier at resolution 8 |
| hour | Hour of day (0 to 23) |
| demand | Actual order count in this hex-hour |
| lag_1 | Demand from the previous hour (engineered feature) |
| predicted_demand | Model's output for this hex-hour |
| polygon | Hexagon boundary coordinates (list of lat/lon tuples) |

---

## 8. API Reference

The FastAPI service exposes three endpoints. Start the server first:

```bash
uvicorn app.main:app --reload --port 8000
```

Then open `http://localhost:8000/docs` for the full interactive Swagger UI
where you can test every endpoint directly in the browser.

**GET /**

Health check. Returns a status message confirming the API is running.

```bash
curl http://localhost:8000/
```

```json
{"status": "ok", "message": "Hex-Demand API is running"}
```

**POST /predict**

Predicts demand for a single hexagon-hour combination.

Request body:

```json
{
    "lag_1": 12.0,
    "hour": 20
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| lag_1 | float | Yes | Orders in the previous hour for this hexagon |
| hour | integer | Yes | Hour of day, 0 to 23 |

Response:

```json
{
    "predicted_demand": 11.43,
    "demand_tier": "High"
}
```

| Field | Description |
|---|---|
| predicted_demand | Predicted order count rounded to 2 decimal places |
| demand_tier | High (10 or above), Medium (5 to 9), or Low (below 5) |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"lag_1": 12.0, "hour": 20}'
```

**POST /predict/batch**

Predicts demand for multiple hexagon-hour records in a single call. Useful
for generating a full city snapshot without making hundreds of individual
requests.

Request body:

```json
{
    "records": [
        {"lag_1": 12.0, "hour": 20},
        {"lag_1": 3.0,  "hour": 4},
        {"lag_1": 7.5,  "hour": 13}
    ]
}
```

Response:

```json
{
    "predictions": [
        {"predicted_demand": 11.43, "demand_tier": "High"},
        {"predicted_demand": 2.87,  "demand_tier": "Low"},
        {"predicted_demand": 6.12,  "demand_tier": "Medium"}
    ]
}
```

**GET /data/summary**

Returns summary statistics from the processed CSV. Useful for a quick sanity
check that the data pipeline ran correctly.

```bash
curl http://localhost:8000/data/summary
```

```json
{
    "total_records"   : 3842,
    "unique_hexagons" : 214,
    "mean_demand"     : 4.37,
    "max_demand"      : 23,
    "hours_covered"   : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
}
```

---

## 9. Streamlit Dashboard

The dashboard reads `artifacts/hex_demand_processed.csv` and renders six
interactive Plotly charts. Make sure you have run `model/train_and_save.py`
before starting the dashboard.

```bash
streamlit run streamlit_app/dashboard.py
```

Open `http://localhost:8501` in your browser.

**Sidebar filters:**

- Hour of Day slider: narrow the analysis to a specific time window
- Minimum predicted demand: hide low-demand hexagons from all charts

**Charts included:**

| Chart | What it shows |
|---|---|
| Distribution of Actual Demand | Histogram confirming right-skewed demand pattern |
| Average Demand by Hour of Day | Bar chart showing which hours are peak and which are quiet |
| Lag-1 Demand vs Current Demand | Scatter with trendline validating the lag feature's predictive signal |
| Actual vs Predicted Demand | Scatter against the perfect-prediction diagonal to assess model accuracy |
| Demand Tier Distribution | Pie chart showing the proportion of High, Medium, and Low zones |
| Top 15 Hexagons by Total Demand | Horizontal bar of the busiest hexagons over the selected time window |
| Demand Heatmap: Hour vs Hexagon | Grid heatmap of predicted demand intensity across hours for the top 20 hexagons |

The dashboard also includes an expandable raw data table at the bottom showing
the filtered records with all relevant columns.

---

## 10. Docker Deployment

Two separate Dockerfiles are provided: one for the FastAPI service and one
for the Streamlit dashboard. The `docker-compose.yml` runs both together.

**Build and run everything with one command:**

```bash
docker-compose up --build
```

- FastAPI API:          `http://localhost:8000`
- FastAPI Swagger docs: `http://localhost:8000/docs`
- Streamlit dashboard:  `http://localhost:8501`

**Run only the API:**

```bash
docker build -t hex-demand-api .
docker run -p 8000:8000 hex-demand-api
```

**Run only the dashboard:**

```bash
docker build -f Dockerfile.streamlit -t hex-demand-dashboard .
docker run -p 8501:8501 hex-demand-dashboard
```

**How the Docker build works:**

Both Dockerfiles copy `requirements.txt` first and run `pip install` before
copying the rest of the source code. This is intentional: Docker caches each
build layer. If only source code changes and requirements stay the same, Docker
skips the slow `pip install` step and reuses the cached layer from the previous
build. This makes iterative development much faster.

Both images also run `python model/train_and_save.py` during the build step.
This bakes the trained model and the processed CSV directly into the image so
the running container is completely self-contained and does not depend on any
external files.

**Stop all running containers:**

```bash
docker-compose down
```

---

## 11. Generated Artifacts

Running `model/train_and_save.py` produces two files in the `artifacts/` folder.
These files are not committed to version control (add `artifacts/` to your
`.gitignore`) because they are generated outputs, not source code.

**artifacts/model.pkl**

The trained Random Forest Regressor saved using joblib. joblib is preferred
over Python's built-in pickle for scikit-learn models because it handles large
numpy arrays more efficiently through memory mapping.

Load it in any Python script with:

```python
import joblib
model = joblib.load('artifacts/model.pkl')
prediction = model.predict([[12.0, 20]])  # [[lag_1, hour]]
```

**artifacts/hex_demand_processed.csv**

The fully processed dataset with all engineered features and model predictions.
This is the file the Streamlit dashboard and the `/data/summary` API endpoint
read from.

Columns: `h3_index`, `hour`, `demand`, `lag_1`, `predicted_demand`, `polygon`

---

## 12. Business Insights and Conclusion

**Key Insights from the project:**

1. Spatial aggregation using H3 indexing effectively identifies demand clusters
   within a city. Raw order coordinates are noisy and hard to act on directly.
   Grouping them into stable hexagonal zones turns that noise into a measurable
   signal.

2. The lag feature captures temporal continuity in customer behaviour. A zone
   that was busy at 7pm is very likely still busy at 8pm. Including the previous
   hour's demand as a feature significantly improves prediction accuracy compared
   to using only hour of day.

3. Random Forest provides accurate and stable predictions for demand density
   without requiring feature scaling or hyperparameter tuning as sensitive as
   gradient boosting methods.

4. Interactive visualisation using Folium enables business teams to make fast
   operational decisions. A coloured city map is far more actionable than a
   table of hexagon IDs and numbers.

5. The model can assist in optimising dark store placement and inventory
   planning. If a set of hexagons consistently shows High demand tier predictions
   during evening hours, placing a dark store in the centre of that cluster
   minimises delivery distances for the highest-volume time window.

---

## 13. Future Enhancements

**Data enrichment:**
- Incorporate weather data. Rain significantly increases grocery delivery demand
  and is a strong predictor that the current model does not see.
- Add festival and public holiday flags. Demand patterns around Durga Puja or
  Diwali differ substantially from ordinary days.
- Add population density per hexagon from census sources as a static baseline
  feature.

**Modelling improvements:**
- Try XGBoost or LightGBM, which consistently outperform scikit-learn's Random
  Forest on structured tabular data in benchmark comparisons.
- Add lag-2 and lag-3 features (demand from two and three hours ago) and a
  rolling 3-hour average for a smoother signal.
- Add H3 neighbour features: the average demand of the 6 surrounding hexagons
  captures spatial autocorrelation and often improves accuracy significantly.

**Engineering and deployment:**
- Build a streaming pipeline using Kafka or AWS Kinesis for real-time predictions
  as orders arrive rather than batch hourly predictions.
- Add model retraining on a weekly schedule using Airflow or Prefect so the
  model stays current as demand patterns shift over time.
- Store hex-demand snapshots in a PostGIS database and serve the Folium map
  tiles from a proper geospatial tile server for production-scale performance.

---

*This project was built as a demonstration of applied spatial data science
for quick-commerce demand forecasting. All data is synthetic and generated
entirely in Python. No real customer or order data was used.*
