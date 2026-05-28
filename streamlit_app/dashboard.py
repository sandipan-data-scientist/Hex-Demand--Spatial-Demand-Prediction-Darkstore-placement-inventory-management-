# Interactive dashboard that reads the processed CSV from artifacts/
# and shows all the charts from the notebook in a clean UI.
# Run with: streamlit run streamlit_app/dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title = "Hex-Demand Dashboard",
    page_icon  = "map",
    layout     = "wide",
)

# ---- Load Data ----

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'hex_demand_processed.csv')

@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        return None
    df = pd.read_csv(CSV_PATH)
    return df

df = load_data()

if df is None:
    st.error("artifacts/hex_demand_processed.csv not found. Run model/train_and_save.py first.")
    st.stop()

# ---- Sidebar Filters ----

st.sidebar.title("Filters")

hour_range = st.sidebar.slider(
    "Hour of Day",
    min_value = int(df['hour'].min()),
    max_value = int(df['hour'].max()),
    value     = (int(df['hour'].min()), int(df['hour'].max())),
)

demand_threshold = st.sidebar.number_input(
    "Minimum predicted demand to show",
    min_value = 0.0,
    max_value = float(df['predicted_demand'].max()),
    value     = 0.0,
    step      = 1.0,
)

# Apply filters
df_filtered = df[
    (df['hour'] >= hour_range[0]) &
    (df['hour'] <= hour_range[1]) &
    (df['predicted_demand'] >= demand_threshold)
].copy()

# Demand tier assignment
def assign_tier(val):
    if val >= 10:
        return "High"
    elif val >= 5:
        return "Medium"
    else:
        return "Low"

df_filtered['demand_tier'] = df_filtered['predicted_demand'].apply(assign_tier)

# ---- Page Header ----

st.title("Hex-Demand: Spatial Demand Forecast Dashboard")
st.caption("Kolkata dark store demand prediction using H3 hexagonal indexing and Random Forest")

st.divider()

# ---- KPI Row ----

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Hex-Hour Records", f"{len(df_filtered):,}")
col2.metric("Unique Hexagons",        f"{df_filtered['h3_index'].nunique()}")
col3.metric("Mean Predicted Demand",  f"{df_filtered['predicted_demand'].mean():.1f}")
col4.metric("Max Predicted Demand",   f"{df_filtered['predicted_demand'].max():.0f}")

st.divider()

# ---- Row 1: Demand Distribution and Hourly Pattern ----

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Distribution of Actual Demand")
    fig_hist = px.histogram(
        df_filtered,
        x         = 'demand',
        nbins     = 35,
        color_discrete_sequence = ['steelblue'],
        labels    = {'demand': 'Orders per Hexagon per Hour'},
        title     = "How often each demand level occurs",
    )
    fig_hist.update_traces(marker_line_width=0.4, marker_line_color='white')
    fig_hist.update_layout(
        yaxis_title  = "Frequency",
        showlegend   = False,
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor= "rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with row1_col2:
    st.subheader("Average Demand by Hour of Day")
    hourly_avg = df_filtered.groupby('hour')['demand'].mean().reset_index()
    fig_bar = px.bar(
        hourly_avg,
        x     = 'hour',
        y     = 'demand',
        color = 'demand',
        color_continuous_scale = 'OrRd',
        labels = {'hour': 'Hour of Day', 'demand': 'Avg Orders'},
        title  = "Which hours drive the most demand",
    )
    fig_bar.update_layout(
        coloraxis_showscale = False,
        plot_bgcolor        = "rgba(0,0,0,0)",
        paper_bgcolor       = "rgba(0,0,0,0)",
        xaxis               = dict(tickmode='linear', dtick=1),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ---- Row 2: Lag Feature Scatter and Actual vs Predicted ----

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("Lag-1 Demand vs Current Demand")
    sample = df_filtered.sample(min(800, len(df_filtered)), random_state=42)
    corr_val = df_filtered[['lag_1', 'demand']].corr().iloc[0, 1]

    fig_scatter = px.scatter(
        sample,
        x       = 'lag_1',
        y       = 'demand',
        opacity = 0.4,
        color_discrete_sequence = ['royalblue'],
        trendline = 'ols',
        labels  = {'lag_1': 'Previous Hour Demand', 'demand': 'Current Demand'},
        title   = f"Pearson r = {corr_val:.3f}  (higher = stronger lag signal)",
    )
    fig_scatter.update_layout(
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with row2_col2:
    st.subheader("Actual vs Predicted Demand")
    sample2 = df_filtered.sample(min(800, len(df_filtered)), random_state=7)

    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(
        x       = sample2['demand'],
        y       = sample2['predicted_demand'],
        mode    = 'markers',
        marker  = dict(color='mediumseagreen', opacity=0.4, size=5),
        name    = 'Predictions',
    ))
    # Perfect prediction line
    max_val = max(sample2['demand'].max(), sample2['predicted_demand'].max())
    fig_avp.add_trace(go.Scatter(
        x    = [0, max_val],
        y    = [0, max_val],
        mode = 'lines',
        line = dict(color='tomato', dash='dash', width=1.5),
        name = 'Perfect Prediction',
    ))
    fig_avp.update_layout(
        xaxis_title   = 'Actual Demand',
        yaxis_title   = 'Predicted Demand',
        title         = "Points near the red line = accurate predictions",
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_avp, use_container_width=True)

# ---- Row 3: Demand Tier Breakdown and Top Hexagons ----

row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    st.subheader("Demand Tier Distribution")
    tier_counts = df_filtered['demand_tier'].value_counts().reset_index()
    tier_counts.columns = ['Tier', 'Count']
    tier_color_map = {'High': '#d73027', 'Medium': '#fc8d59', 'Low': '#1a9850'}

    fig_pie = px.pie(
        tier_counts,
        names  = 'Tier',
        values = 'Count',
        color  = 'Tier',
        color_discrete_map = tier_color_map,
        title  = "Share of hex-hour records by demand tier",
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with row3_col2:
    st.subheader("Top 15 Hexagons by Total Predicted Demand")
    top_hex = (
        df_filtered.groupby('h3_index')['predicted_demand']
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    top_hex.columns = ['H3 Index', 'Total Predicted Demand']
    top_hex['H3 Short'] = top_hex['H3 Index'].str[-6:]  # last 6 chars for readability

    fig_top = px.bar(
        top_hex,
        x     = 'Total Predicted Demand',
        y     = 'H3 Short',
        orientation = 'h',
        color = 'Total Predicted Demand',
        color_continuous_scale = 'Blues',
        labels = {'H3 Short': 'Hexagon (last 6 chars of ID)'},
        title  = "Which hexagons accumulate the most demand",
    )
    fig_top.update_layout(
        yaxis              = dict(autorange='reversed'),
        coloraxis_showscale= False,
        plot_bgcolor       = "rgba(0,0,0,0)",
        paper_bgcolor      = "rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_top, use_container_width=True)

# ---- Row 4: Predicted Demand Heatmap by Hour ----

st.subheader("Predicted Demand Heatmap: Hour vs Hexagon Cluster")
st.caption("Each cell shows average predicted demand for a given hour. Darker = higher demand.")

# Use top 20 hexagons for the heatmap so it stays readable
top20_hex = (
    df_filtered.groupby('h3_index')['demand']
    .sum()
    .sort_values(ascending=False)
    .head(20)
    .index
    .tolist()
)

heatmap_df = df_filtered[df_filtered['h3_index'].isin(top20_hex)].copy()
heatmap_df['h3_short'] = heatmap_df['h3_index'].str[-6:]

pivot = heatmap_df.pivot_table(
    index   = 'h3_short',
    columns = 'hour',
    values  = 'predicted_demand',
    aggfunc = 'mean',
).fillna(0)

fig_hmap = px.imshow(
    pivot,
    color_continuous_scale = 'YlOrRd',
    aspect                 = 'auto',
    labels                 = dict(x='Hour of Day', y='Hexagon', color='Predicted Demand'),
    title                  = "Demand intensity across hours for top 20 hexagons",
)
fig_hmap.update_layout(
    paper_bgcolor = "rgba(0,0,0,0)",
    height        = 450,
)
st.plotly_chart(fig_hmap, use_container_width=True)

# ---- Row 5: Raw Data Table (collapsible) ----

with st.expander("View filtered raw data"):
    st.dataframe(
        df_filtered[['h3_index', 'hour', 'demand', 'lag_1', 'predicted_demand', 'demand_tier']],
        use_container_width = True,
        height              = 300,
    )
    st.caption(f"Showing {len(df_filtered)} rows after filters.")

    # streamlit_app/dashboard.py  (map section — add after your existing charts)
#
# This section builds a Folium hexagon map from the processed CSV and
# renders it live inside Streamlit using streamlit-folium.
# The map colours each H3 hexagon by its predicted demand tier.

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import h3
import os

# ---- Load data ----
# Re-use the cached df from the top of your dashboard if it is already loaded.
# If you are adding this as a standalone section, load it here.

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'hex_demand_processed.csv')

@st.cache_data
def load_hex_data():
    if not os.path.exists(CSV_PATH):
        return None
    df = pd.read_csv(CSV_PATH)
    return df

df = load_hex_data()

if df is None:
    st.error("artifacts/hex_demand_processed.csv not found. Run model/train_and_save.py first.")
    st.stop()

# ---- Section header ----

st.divider()
st.subheader("Predicted Demand Heatmap: Hexagonal City Grid")
st.caption(
    "Each hexagon on the map represents one H3 spatial cell at resolution 8 "
    "(approximately 0.74 sq km). Fill colour and opacity reflect predicted "
    "demand intensity. Click any hexagon to see its exact predicted demand value."
)

# ---- Map controls in a horizontal row ----

map_col1, map_col2, map_col3 = st.columns(3)

with map_col1:
    selected_hour = st.selectbox(
        "Filter by Hour of Day",
        options    = ["All Hours"] + sorted(df['hour'].unique().tolist()),
        index      = 0,
        help       = "Show only hexagons active during a specific hour."
    )

with map_col2:
    color_scheme = st.selectbox(
        "Colour Scheme",
        options = ["Red intensity", "Traffic light tiers", "Blue gradient"],
        index   = 0,
        help    = "How hexagons are coloured on the map."
    )

with map_col3:
    show_heatmap_layer = st.checkbox(
        "Overlay raw HeatMap layer",
        value = False,
        help  = "Adds a kernel density heatmap on top of the hexagons, "
                "useful for seeing where order points are densest."
    )

# ---- Filter data based on controls ----

if selected_hour == "All Hours":
    df_map = df.copy()
else:
    df_map = df[df['hour'] == int(selected_hour)].copy()

# If multiple hours are included, take each hexagon's mean predicted demand
# so there is one value per hexagon on the map
df_agg = (
    df_map.groupby('h3_index')
    .agg(
        predicted_demand = ('predicted_demand', 'mean'),
        actual_demand    = ('demand',           'mean'),
        hours_seen       = ('hour',             'nunique'),
    )
    .reset_index()
)

# ---- Demand tier assignment ----
# Thresholds are percentile-based on the aggregated snapshot
# so the colour spread adapts to the current filter selection

p33 = df_agg['predicted_demand'].quantile(0.33)
p67 = df_agg['predicted_demand'].quantile(0.67)

def assign_tier(val):
    if val >= p67:
        return 'High'
    elif val >= p33:
        return 'Medium'
    else:
        return 'Low'

df_agg['demand_tier'] = df_agg['predicted_demand'].apply(assign_tier)

# ---- Colour logic per scheme ----

max_pred = df_agg['predicted_demand'].max()
# Protect against divide-by-zero if all predictions are the same value
if max_pred == 0:
    max_pred = 1.0

def get_hex_color_and_opacity(row, scheme):
    tier = row['demand_tier']
    norm = row['predicted_demand'] / max_pred  # 0.0 to 1.0

    if scheme == "Red intensity":
        # Single red hue, opacity proportional to demand
        # Low opacity = faint red, high opacity = solid red
        return '#cc2200', max(0.15, norm)

    elif scheme == "Traffic light tiers":
        # Three-colour system matching the notebook's map legend
        color_map = {'High': '#d73027', 'Medium': '#fc8d59', 'Low': '#1a9850'}
        return color_map[tier], 0.65

    elif scheme == "Blue gradient":
        # Dark blue for high demand, light blue for low
        # We interpolate between two hex colours manually
        if norm >= 0.67:
            return '#08306b', 0.75   # darkest blue
        elif norm >= 0.33:
            return '#2171b5', 0.60   # mid blue
        else:
            return '#9ecae1', 0.45   # light blue

    return '#cc2200', norm           # fallback


# ---- Get hexagon boundary coordinates ----
# h3.cell_to_boundary returns list of (lat, lon) tuples (h3-py v4 API)
# h3.h3_to_geo_boundary returns same in v3 API
# We handle both with a try/except so this works on either version

def get_boundary(h3_idx):
    try:
        return h3.cell_to_boundary(h3_idx)
    except AttributeError:
        return h3.h3_to_geo_boundary(h3_idx, geo_json=False)

def get_center(h3_idx):
    try:
        return h3.cell_to_latlng(h3_idx)
    except AttributeError:
        return h3.h3_to_geo(h3_idx)


# ---- Build the Folium map ----

# Center on Kolkata (matching the notebook)
KOLKATA_LAT = 22.5726
KOLKATA_LON = 88.3639

m = folium.Map(
    location   = [KOLKATA_LAT, KOLKATA_LON],
    zoom_start = 12,
    tiles      = 'CartoDB dark_matter',   # dark background makes coloured hexagons pop
)

# Draw each hexagon as a polygon
for _, row in df_agg.iterrows():
    h3_idx   = row['h3_index']
    color, opacity = get_hex_color_and_opacity(row, color_scheme)

    boundary = get_boundary(h3_idx)
    # Folium Polygon expects [[lat, lon], ...]
    polygon_coords = [[lat, lon] for lat, lon in boundary]

    # Build a clean popup with all relevant info
    # folium.Popup with html=True lets us format it nicely
    popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 13px; min-width: 180px;">
            <b>H3 Index</b><br>
            <code style="font-size:11px">{h3_idx}</code><br><br>
            <b>Predicted Demand:</b> {row['predicted_demand']:.1f} orders<br>
            <b>Actual Demand:</b>    {row['actual_demand']:.1f} orders<br>
            <b>Demand Tier:</b>      {row['demand_tier']}<br>
            <b>Hours Covered:</b>    {int(row['hours_seen'])}
        </div>
    """

    folium.Polygon(
        locations    = polygon_coords,
        color        = color,
        weight       = 0.8,               # border thickness
        fill         = True,
        fill_color   = color,
        fill_opacity = opacity,
        popup        = folium.Popup(popup_html, max_width=250),
        tooltip      = f"{row['demand_tier']} demand: {row['predicted_demand']:.1f} orders",
    ).add_to(m)


# ---- Optional raw HeatMap layer ----
# This uses the actual order locations (approximated from hex centres)
# to draw a kernel density heatmap on top of the hexagons
# It gives a different visual: a smooth heat gradient vs discrete hexagons

if show_heatmap_layer:
    heat_data = []
    for _, row in df_agg.iterrows():
        center = get_center(row['h3_index'])
        lat, lon = center[0], center[1]
        weight = row['predicted_demand']
        heat_data.append([lat, lon, weight])

    HeatMap(
        heat_data,
        min_opacity = 0.2,
        max_val     = max_pred,
        radius      = 18,
        blur        = 20,
        gradient    = {0.4: 'blue', 0.65: 'lime', 0.85: 'orange', 1.0: 'red'},
    ).add_to(m)


# ---- Map legend (injected as HTML into the Folium map) ----
# This is the same approach as the notebook's standalone map

if color_scheme == "Traffic light tiers":
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px;
        background: rgba(20,20,30,0.92);
        color: white; padding: 12px 16px;
        border-radius: 8px; font-family: Arial, sans-serif;
        font-size: 13px; z-index: 9999;
        border: 1px solid #555;
    ">
        <b>Demand Tier</b><br>
        <span style="color:#d73027">&#9632;</span> High<br>
        <span style="color:#fc8d59">&#9632;</span> Medium<br>
        <span style="color:#1a9850">&#9632;</span> Low
    </div>
    """
elif color_scheme == "Red intensity":
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px;
        background: rgba(20,20,30,0.92);
        color: white; padding: 12px 16px;
        border-radius: 8px; font-family: Arial, sans-serif;
        font-size: 13px; z-index: 9999;
        border: 1px solid #555;
    ">
        <b>Demand Intensity</b><br>
        <span style="color:#cc2200; opacity:1.0">&#9632;</span> High<br>
        <span style="color:#cc2200; opacity:0.5">&#9632;</span> Medium<br>
        <span style="color:#cc2200; opacity:0.2">&#9632;</span> Low
    </div>
    """
else:
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px;
        background: rgba(20,20,30,0.92);
        color: white; padding: 12px 16px;
        border-radius: 8px; font-family: Arial, sans-serif;
        font-size: 13px; z-index: 9999;
        border: 1px solid #555;
    ">
        <b>Demand Level</b><br>
        <span style="color:#08306b">&#9632;</span> High<br>
        <span style="color:#2171b5">&#9632;</span> Medium<br>
        <span style="color:#9ecae1">&#9632;</span> Low
    </div>
    """

m.get_root().html.add_child(folium.Element(legend_html))


# ---- Render the map in Streamlit ----
# st_folium streams the Folium map object directly into the Streamlit page.
# returned_objects captures what the user clicks on, enabling reactive behaviour.
# use_container_width=True makes the map fill the full column width.

map_output = st_folium(
    m,
    width              = "100%",
    height             = 550,
    returned_objects   = ["last_object_clicked_popup"],
    use_container_width= True,
)

# ---- Show clicked hexagon details below the map ----
# When the user clicks a hexagon, the popup HTML comes back in map_output.
# We surface it as a clean info box below the map as well.

if map_output and map_output.get("last_object_clicked_popup"):
    st.info(
        "Hexagon clicked. See the popup on the map above for full details, "
        "or filter by hour using the controls above to isolate that zone."
    )


# ---- Summary stats for current map view ----

st.caption(f"Showing {len(df_agg)} hexagons on the map.")

summary_col1, summary_col2, summary_col3 = st.columns(3)

high_count   = (df_agg['demand_tier'] == 'High').sum()
medium_count = (df_agg['demand_tier'] == 'Medium').sum()
low_count    = (df_agg['demand_tier'] == 'Low').sum()

summary_col1.metric("High Demand Zones",   high_count,   help="Hexagons above the 67th percentile of predicted demand")
summary_col2.metric("Medium Demand Zones", medium_count, help="Hexagons between the 33rd and 67th percentile")
summary_col3.metric("Low Demand Zones",    low_count,    help="Hexagons below the 33rd percentile of predicted demand")