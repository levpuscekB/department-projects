import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px  
import numpy as np

# ---- CONFIG ----
st.set_page_config(page_title="Institute Projects", layout="wide")
st.title("📊 JSI Long-term Projects Plan")
st.markdown("""
This dashboard visualizes institute project activities using data from a [shared Google Sheet](https://docs.google.com/spreadsheets/d/1qBKlVTh3_U3Mk6OsBPRnTjrJEmg9VC2KU6BjlIj1yUo/edit?usp=sharing).
Projects are displayed in a Cost vs. Time plot, with each department represented by an unique color.  
You can filter by specific departments.  
If points overlap, increase the separation radius (left sidebar) to spread them apart.  
Below the plot, you can view the raw data table.
If any errors arise, please contact [blaz.levpuscek@ijs.si](mailto:blaz.levpuscek@ijs.si).
""")


def parse_euro_number(val):
    """
    Parses a number with European decimal comma or scientific notation.
    No thousands separators are expected.
    Examples:
        '1000,50' -> 1000.5
        '1e6'     -> 1000000.0
        '1000'    -> 1000.0
    """
    if pd.isnull(val):
        return float('nan')
    s = str(val).strip().replace(' ', '')
    if ',' in s:
        s = s.replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return float('nan')



# ---- SHEET CONFIG ----
SHEET_ID = "1qBKlVTh3_U3Mk6OsBPRnTjrJEmg9VC2KU6BjlIj1yUo"
SHEET_NAME = "AllData"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

if st.button("🔄 Update plot from Google Sheet"):
    st.session_state["refresh"] = True

@st.cache_data(ttl=0)
def load_data():
    df = pd.read_csv(CSV_URL)
    df = df.dropna(subset=["Project", "Time [years]", "Cost [k€]", "Department"])
    df["Time (years)"] = pd.to_numeric(df["Time [years]"], errors="coerce")
    df["Cost (k€)"] = df["Cost [k€]"].apply(parse_euro_number)
    df["Cost (€)"] = df["Cost (k€)"] * 1000
    return df


if st.session_state.get("refresh", False):
    st.cache_data.clear()
    st.session_state["refresh"] = False

data = load_data()

# ---- FILTERS ----
departments = data["Department"].unique().tolist()
selected_departments = st.multiselect("Filter by Department", departments, default=departments)
filtered = data[data["Department"].isin(selected_departments)]

# ---- SLIDER FOR SEPARATION RADIUS ----
radius = st.sidebar.slider(
    "Department separation radius (log-units, higher = wider spread)", 
    min_value=0.00, max_value=0.05, value=0.01, step=0.005
)

# ---- COLOR MAP ----
department_list = list(filtered["Department"].unique())
color_palette = px.colors.qualitative.Light24
color_map = {dept: color_palette[i % len(color_palette)] for i, dept in enumerate(department_list)}

# ---- DYNAMIC DEPARTMENT ANGLES ----
n_dept = len(department_list)
angles = {dept: 2 * np.pi * i / n_dept for i, dept in enumerate(sorted(department_list))}

# ---- MAIN PROCESSING ----
fig = go.Figure()

for dept in department_list:
    df_dept = filtered[filtered["Department"] == dept]
    if df_dept.empty:
        continue

    log_x = np.log10(df_dept["Time (years)"].values)
    log_y = np.log10(df_dept["Cost (€)"].values)
    projects = df_dept["Project"].values
    descriptions = df_dept["Description"].values

    # Apply department-specific offset in log-log space
    angle = angles[dept]
    dx = radius * np.cos(angle)
    dy = radius * np.sin(angle)
    offset_log_x = log_x + dx
    offset_log_y = log_y + dy

    # Combine close points (single-linkage clustering in log-log+offset space)
    n_points = len(offset_log_x)
    used = np.zeros(n_points, dtype=bool)
    combined_points = []

    for i in range(n_points):
        if used[i]:
            continue
        # Start new group with point i
        group = [i]
        for j in range(i+1, n_points):
            if used[j]:
                continue
            dist = np.sqrt((offset_log_x[i] - offset_log_x[j])**2 + (offset_log_y[i] - offset_log_y[j])**2)
            if dist < radius:
                group.append(j)
        for idx in group:
            used[idx] = True
        # Combine group's properties
        mean_log_x = np.mean(offset_log_x[group])
        mean_log_y = np.mean(offset_log_y[group])
        group_projects = [projects[idx] for idx in group]
        group_descriptions = [descriptions[idx] for idx in group]
        count = len(group)
        hover = "<br>".join([f"<b>{p}</b>: {d}" for p, d in zip(group_projects, group_descriptions)])
        combined_points.append({
            "log_x": mean_log_x,
            "log_y": mean_log_y,
            "size": 14 + 6 * (count-1),
            "hover": hover,
            "count": count
        })

    # Scatter for department
    if combined_points:
        fig.add_trace(go.Scatter(
            x=[10 ** pt["log_x"] for pt in combined_points],
            y=[10 ** pt["log_y"] for pt in combined_points],
            mode="markers",
            name=dept,
            marker=dict(
                size=[pt["size"] for pt in combined_points],
                color=color_map.get(dept, "#cccccc"),
                line=dict(width=1, color="DarkSlateGrey"),
                opacity=0.8
            ),
            hovertext=[
                f"<b>{dept}</b> ({pt['count']} project{'s' if pt['count'] > 1 else ''}):<br>{pt['hover']}" 
                for pt in combined_points
            ],
            hoverinfo="text",
            legendgroup=dept,
            showlegend=True
        ))

fig.update_layout(
    xaxis=dict(title="Time (years)", type="log"),
    yaxis=dict(title="Cost (€)", type="log"),
    height=700
)

if len(fig.data) == 0:
    st.warning("⚠️ No data available for the selected filters.")
else:
    st.plotly_chart(fig, use_container_width=True)

columns_to_show = [col for col in ["Project", "Time [years]", "Cost [k€]", "Department", "Description", "Longer description"] if col in filtered.columns]
with st.expander("🔍 View Raw Data Table", expanded=True):
    st.dataframe(filtered[columns_to_show])

