# streamlit_app/dashboard.py
# Fully rewritten to remove plotly dependency.
# All charts use native Streamlit chart functions which are built-in
# and require no additional packages beyond streamlit itself.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import h3
import os

st.set_page_config(
    page_title = "Hex-Demand Dashboard",
    layout     = "wide",
)

# ---- Load data ----

CSV_PATH = os.path.join('artifacts', 'hex_demand_processed.csv')

@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        return None
    df = pd.read_csv(CSV_PATH)
    return df

df = load_data()

if df is None:
    st.error(
        "artifacts/hex_demand_processed.csv not found. "
        "Run model/train_and_save.py first and commit the CSV to the repo."
    )
    st.stop()

# ---- Sidebar filters ----

st.sidebar.title("Filters")

hour_range = st.sidebar.slider(
    "Hour of Day",
    min_value = int(df['hour'].min()),
    max_value = int(df['hour'].max()),
    value     = (int(df['hour'].min()), int(df['hour'].max())),
)

demand_threshold = st.sidebar.number_input(
    "Minimum predicted demand",
    min_value = 0.0,
    max_value = float(df['predicted_demand'].max()),
    value     = 0.0,
    step      = 1.0,
)

df_filtered = df[
    (df['hour'] >= hour_range[0]) &
    (df['hour'] <= hour_range[1]) &
    (df['predicted_demand'] >= demand_threshold)
].copy()

def assign_tier(val, p33, p67):
    if val >= p67:
        return 'High'
    elif val >= p33:
        return 'Medium'
    else:
        return 'Low'

p33 = df_filtered['predicted_demand'].quantile(0.33)
p67 = df_filtered['predicted_demand'].quantile(0.67)
df_filtered['demand_tier'] = df_filtered['predicted_demand'].apply(
    lambda v: assign_tier(v, p33, p67)
)

# ---- Header ----

st.title("Hex-Demand: Spatial Demand Forecast Dashboard")
st.caption(
    "Kolkata dark store demand prediction using H3 hexagonal indexing "
    "and Random Forest."
)

st.divider()

# ---- KPI row ----

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Records",          f"{len(df_filtered):,}")
k2.metric("Unique Hexagons",        f"{df_filtered['h3_index'].nunique()}")
k3.metric("Mean Predicted Demand",  f"{df_filtered['predicted_demand'].mean():.1f}")
k4.metric("Max Predicted Demand",   f"{df_filtered['predicted_demand'].max():.0f}")

st.divider()

# ---- Row 1: Demand distribution and hourly pattern ----
# Using matplotlib here instead of plotly because matplotlib is already
# a dependency of scikit-learn and is always available on Streamlit Cloud

r1c1, r1c2 = st.columns(2)

with r1c1:
    st.subheader("Distribution of Actual Demand")
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.hist(
        df_filtered['demand'],
        bins       = 35,
        color      = 'steelblue',
        edgecolor  = 'white',
        linewidth  = 0.4,
    )
    ax1.axvline(
        df_filtered['demand'].mean(),
        color     = 'tomato',
        linestyle = '--',
        linewidth = 1.5,
        label     = f"Mean = {df_filtered['demand'].mean():.1f}"
    )
    ax1.set_xlabel('Orders per Hexagon per Hour')
    ax1.set_ylabel('Frequency')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

with r1c2:
    st.subheader("Average Demand by Hour of Day")
    hourly_avg = (
        df_filtered.groupby('hour')['demand']
        .mean()
        .reset_index()
        .rename(columns={'demand': 'avg_demand'})
        .set_index('hour')
    )
    # st.bar_chart natively uses the dataframe index as x-axis
    st.bar_chart(hourly_avg, use_container_width=True)

# ---- Row 2: Lag scatter and actual vs predicted ----

r2c1, r2c2 = st.columns(2)

with r2c1:
    st.subheader("Lag-1 Demand vs Current Demand")
    corr_val = df_filtered[['lag_1', 'demand']].corr().iloc[0, 1]
    st.caption(f"Pearson r = {corr_val:.3f} — higher means the lag feature carries stronger signal.")

    sample = df_filtered.sample(min(800, len(df_filtered)), random_state=42)
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.scatter(
        sample['lag_1'],
        sample['demand'],
        alpha = 0.35,
        s     = 12,
        color = 'royalblue',
    )
    # Trend line using numpy linear regression
    m, b = np.polyfit(sample['lag_1'], sample['demand'], 1)
    x_line = np.linspace(sample['lag_1'].min(), sample['lag_1'].max(), 100)
    ax2.plot(x_line, m * x_line + b, color='tomato', linewidth=1.8, label='Trend')
    ax2.set_xlabel('Previous Hour Demand (lag_1)')
    ax2.set_ylabel('Current Demand')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

with r2c2:
    st.subheader("Actual vs Predicted Demand")
    st.caption("Points near the diagonal line mean accurate predictions.")

    sample2 = df_filtered.sample(min(800, len(df_filtered)), random_state=7)
    fig3, ax3 = plt.subplots(figsize=(6, 3.5))
    ax3.scatter(
        sample2['demand'],
        sample2['predicted_demand'],
        alpha = 0.35,
        s     = 12,
        color = 'mediumseagreen',
    )
    max_val = max(sample2['demand'].max(), sample2['predicted_demand'].max())
    ax3.plot(
        [0, max_val], [0, max_val],
        color     = 'tomato',
        linestyle = '--',
        linewidth = 1.5,
        label     = 'Perfect prediction',
    )
    ax3.set_xlabel('Actual Demand')
    ax3.set_ylabel('Predicted Demand')
    ax3.legend(fontsize=9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

# ---- Row 3: Tier breakdown and top hexagons ----

r3c1, r3c2 = st.columns(2)

with r3c1:
    st.subheader("Demand Tier Distribution")
    tier_counts = (
        df_filtered['demand_tier']
        .value_counts()
        .reindex(['High', 'Medium', 'Low'])
        .fillna(0)
        .astype(int)
    )
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    colors = ['#d73027', '#fc8d59', '#1a9850']
    ax4.pie(
        tier_counts.values,
        labels    = tier_counts.index,
        colors    = colors,
        autopct   = '%1.1f%%',
        startangle= 140,
        textprops = {'fontsize': 11},
    )
    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

with r3c2:
    st.subheader("Top 15 Hexagons by Total Predicted Demand")
    top_hex = (
        df_filtered.groupby('h3_index')['predicted_demand']
        .sum()
        .sort_values(ascending=True)
        .tail(15)
    )
    # Shorten h3 index to last 6 chars for readability on the axis
    top_hex.index = top_hex.index.str[-6:]
    top_hex = top_hex.to_frame(name='Total Predicted Demand')

    fig5, ax5 = plt.subplots(figsize=(6, 4))
    ax5.barh(
        top_hex.index,
        top_hex['Total Predicted Demand'],
        color     = 'royalblue',
        edgecolor = 'white',
        linewidth = 0.4,
    )
    ax5.set_xlabel('Total Predicted Demand')
    ax5.set_ylabel('Hexagon (last 6 chars)')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    fig5.tight_layout()
    st.pyplot(fig5, use_container_width=True)
    plt.close(fig5)

# ---- Row 4: Demand heatmap by hour ----

st.subheader("Predicted Demand Heatmap: Hour vs Hexagon")
st.caption("Darker cells mean higher average predicted demand for that hexagon during that hour.")

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

fig6, ax6 = plt.subplots(figsize=(14, 6))
im = ax6.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
ax6.set_xticks(range(len(pivot.columns)))
ax6.set_xticklabels(pivot.columns, fontsize=8)
ax6.set_yticks(range(len(pivot.index)))
ax6.set_yticklabels(pivot.index, fontsize=8)
ax6.set_xlabel('Hour of Day')
ax6.set_ylabel('Hexagon (last 6 chars)')
plt.colorbar(im, ax=ax6, label='Predicted Demand')
fig6.tight_layout()
st.pyplot(fig6, use_container_width=True)
plt.close(fig6)

# ---- Folium map ----

st.divider()
st.subheader("Predicted Demand Heatmap: Hexagonal City Grid")
st.caption(
    "Each hexagon represents one H3 spatial cell at resolution 8 "
    "(approximately 0.74 sq km). Click any hexagon for details."
)

map_col1, map_col2, map_col3 = st.columns(3)

with map_col1:
    selected_hour = st.selectbox(
        "Filter Map by Hour",
        options = ["All Hours"] + sorted(df['hour'].unique().tolist()),
        index   = 0,
    )
with map_col2:
    color_scheme = st.selectbox(
        "Colour Scheme",
        options = ["Traffic light tiers", "Red intensity", "Blue gradient"],
        index   = 0,
    )
with map_col3:
    show_heatmap_layer = st.checkbox("Overlay raw HeatMap layer", value=False)

if selected_hour == "All Hours":
    df_map = df.copy()
else:
    df_map = df[df['hour'] == int(selected_hour)].copy()

df_agg = (
    df_map.groupby('h3_index')
    .agg(
        predicted_demand = ('predicted_demand', 'mean'),
        actual_demand    = ('demand',           'mean'),
        hours_seen       = ('hour',             'nunique'),
    )
    .reset_index()
)

p33m = df_agg['predicted_demand'].quantile(0.33)
p67m = df_agg['predicted_demand'].quantile(0.67)
df_agg['demand_tier'] = df_agg['predicted_demand'].apply(
    lambda v: assign_tier(v, p33m, p67m)
)

max_pred = df_agg['predicted_demand'].max()
if max_pred == 0:
    max_pred = 1.0

def get_hex_color_and_opacity(row, scheme):
    tier = row['demand_tier']
    norm = row['predicted_demand'] / max_pred
    if scheme == "Red intensity":
        return '#cc2200', max(0.15, norm)
    elif scheme == "Traffic light tiers":
        return {'High': '#d73027', 'Medium': '#fc8d59', 'Low': '#1a9850'}[tier], 0.65
    else:
        if norm >= 0.67:
            return '#08306b', 0.75
        elif norm >= 0.33:
            return '#2171b5', 0.60
        else:
            return '#9ecae1', 0.45

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

KOLKATA_LAT = 22.5726
KOLKATA_LON = 88.3639

m = folium.Map(
    location   = [KOLKATA_LAT, KOLKATA_LON],
    zoom_start = 12,
    tiles      = 'CartoDB dark_matter',
)

for _, row in df_agg.iterrows():
    color, opacity = get_hex_color_and_opacity(row, color_scheme)
    boundary       = get_boundary(row['h3_index'])
    polygon_coords = [[lat, lon] for lat, lon in boundary]

    popup_html = f"""
        <div style="font-family:Arial,sans-serif;font-size:13px;min-width:180px">
            <b>H3 Index</b><br>
            <code style="font-size:11px">{row['h3_index']}</code><br><br>
            <b>Predicted Demand:</b> {row['predicted_demand']:.1f}<br>
            <b>Actual Demand:</b> {row['actual_demand']:.1f}<br>
            <b>Tier:</b> {row['demand_tier']}
        </div>
    """
    folium.Polygon(
        locations    = polygon_coords,
        color        = color,
        weight       = 0.8,
        fill         = True,
        fill_color   = color,
        fill_opacity = opacity,
        popup        = folium.Popup(popup_html, max_width=250),
        tooltip      = f"{row['demand_tier']}: {row['predicted_demand']:.1f} orders",
    ).add_to(m)

if show_heatmap_layer:
    heat_data = []
    for _, row in df_agg.iterrows():
        c = get_center(row['h3_index'])
        heat_data.append([c[0], c[1], row['predicted_demand']])
    HeatMap(
        heat_data,
        min_opacity = 0.2,
        max_val     = max_pred,
        radius      = 18,
        blur        = 20,
        gradient    = {0.4: 'blue', 0.65: 'lime', 0.85: 'orange', 1.0: 'red'},
    ).add_to(m)

legend_colors = {
    "Traffic light tiers": [('#d73027','High'), ('#fc8d59','Medium'), ('#1a9850','Low')],
    "Red intensity":       [('#cc2200','High'), ('#cc220088','Medium'), ('#cc220033','Low')],
    "Blue gradient":       [('#08306b','High'), ('#2171b5','Medium'),   ('#9ecae1','Low')],
}
legend_rows = "".join(
    f'<span style="color:{c}">&#9632;</span> {label}<br>'
    for c, label in legend_colors[color_scheme]
)
legend_html = f"""
<div style="position:fixed;bottom:30px;left:30px;
    background:rgba(20,20,30,0.92);color:white;
    padding:12px 16px;border-radius:8px;
    font-family:Arial,sans-serif;font-size:13px;
    z-index:9999;border:1px solid #555">
    <b>Demand Tier</b><br>{legend_rows}
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(
    m,
    width               = "100%",
    height              = 550,
    returned_objects    = ["last_object_clicked_popup"],
    use_container_width = True,
)

sc1, sc2, sc3 = st.columns(3)
sc1.metric("High Zones",   (df_agg['demand_tier'] == 'High').sum())
sc2.metric("Medium Zones", (df_agg['demand_tier'] == 'Medium').sum())
sc3.metric("Low Zones",    (df_agg['demand_tier'] == 'Low').sum())

# ---- Raw data table ----

with st.expander("View filtered raw data"):
    st.dataframe(
        df_filtered[['h3_index','hour','demand','lag_1','predicted_demand','demand_tier']],
        use_container_width = True,
        height              = 300,
    )
    st.caption(f"Showing {len(df_filtered)} rows.")