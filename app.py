import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="FPL Mini-League Dashboard",
    page_icon="⚽",
    layout="wide",
)

DATA_PATH = "data/fpl_history.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Season"] = df["Season"].astype(str)
    df["Gameweek"] = pd.to_numeric(df["Gameweek"], errors="coerce")
    df = df.dropna(subset=["Gameweek"]).copy()
    df["Gameweek"] = df["Gameweek"].astype(int)
    for col in [
        "points",
        "actualGameweekPoints",
        "total_points",
        "overall_rank",
        "points_on_bench",
        "event_transfers_cost",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find `{DATA_PATH}`. Add your CSV there and refresh.")
    st.stop()

if df.empty:
    st.warning("Your dataset is empty.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Dashboard Filters")

seasons = sorted(df["Season"].unique())
selected_season = st.sidebar.selectbox(
    "Season",
    options=seasons,
    index=len(seasons) - 1,
)

season_df = df[df["Season"] == selected_season].copy()
min_gw, max_gw = int(season_df["Gameweek"].min()), int(season_df["Gameweek"].max())

if min_gw < max_gw:
    gw_min, gw_max = st.sidebar.slider(
        "Gameweek Range",
        min_value=min_gw,
        max_value=max_gw,
        value=(min_gw, max_gw),
    )
else:
    gw_min = gw_max = min_gw

all_managers = sorted(season_df["playerName"].dropna().unique())
selected_managers = st.sidebar.multiselect(
    "Managers (leave empty for all)",
    options=all_managers,
    default=[],
)

filtered_df = season_df[
    (season_df["Gameweek"] >= gw_min) & (season_df["Gameweek"] <= gw_max)
].copy()
if selected_managers:
    filtered_df = filtered_df[filtered_df["playerName"].isin(selected_managers)]

if filtered_df.empty:
    st.info("No rows match the current filter selection.")
    st.stop()

# --- Core Standings & Deltas at Selected End Gameweek ---
latest_gw = int(filtered_df["Gameweek"].max())
prev_gw = latest_gw - 1 if latest_gw > gw_min else None

standings = (
    filtered_df[filtered_df["Gameweek"] == latest_gw]
    .sort_values("total_points", ascending=False)
    .reset_index(drop=True)
)
standings["Pos"] = standings["total_points"].rank(ascending=False, method="min").astype(int)
standings["Gap to Above"] = standings["total_points"].shift(1) - standings["total_points"]

if prev_gw is not None and not filtered_df[filtered_df["Gameweek"] == prev_gw].empty:
    prev_pos = (
        filtered_df[filtered_df["Gameweek"] == prev_gw]
        .set_index("playerName")["total_points"]
        .rank(ascending=False, method="min")
    )
    standings["Change in position"] = (
        standings["playerName"].map(prev_pos) - standings["Pos"]
    ).fillna(0).astype(int)
else:
    standings["Change in position"] = 0

bench_season = (
    filtered_df.groupby("playerName", as_index=False)["points_on_bench"]
    .sum()
    .rename(columns={"points_on_bench": "Season Bench Pts"})
)
standings = standings.merge(bench_season, on="playerName", how="left")

# --- Header & KPIs ---
st.title(f"⚽ FPL Mini-League Dashboard — {selected_season}")
st.caption(f"Showing Gameweeks {gw_min}–{gw_max} (Latest: GW {latest_gw})")

leader = standings.iloc[0]
second_gap = int(standings.iloc[1]["Gap to Above"]) if len(standings) > 1 and pd.notna(standings.iloc[1]["Gap to Above"]) else 0
top_bench = standings.sort_values("Season Bench Pts", ascending=False).iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("League Leader", f"{leader['playerName']}", f"{int(leader['total_points'])} pts")
k2.metric("#1 vs #2 Gap", f"{second_gap} pts", standings.iloc[1]["playerName"] if len(standings) > 1 else "")
k3.metric("Avg GW Score", f"{filtered_df['actualGameweekPoints'].mean():.1f} pts")
k4.metric(
    "Most Bench Pts",
    f"{top_bench['playerName']}",
    f"{int(top_bench['Season Bench Pts'])} pts",
)

st.divider()

# --- Tabs for Visualisations & Matrices ---
tab_overview, tab_matrices, tab_raw = st.tabs(
    ["📈 Standings & Charts", "🔥 GW & Position Matrices", "📄 Data Table"]
)

with tab_overview:
    col_left, col_right = st.columns([1.2, 1.0])

    with col_left:
        st.subheader("Cumulative Points Trajectory")
        traj_pivot = filtered_df.pivot(
            index="Gameweek", columns="playerName", values="total_points"
        )
        st.line_chart(traj_pivot, height=360)

    with col_right:
        st.subheader(f"GW {latest_gw} Standings")
        display_cols = [
            "Pos",
            "playerName",
            "total_points",
            "actualGameweekPoints",
            "Gap to Above",
            "Change in position",
        ]
        st.dataframe(
            standings[display_cols].rename(
                columns={
                    "playerName": "Manager",
                    "total_points": "Total Pts",
                    "actualGameweekPoints": f"GW {latest_gw} Pts",
                    "Change in position": "Δ Pos",
                }
            ),
            hide_index=True,
            use_container_width=True,
            height=360,
        )

    st.subheader("Cumulative Points Left on Bench")
    bench_chart_df = (
        standings.sort_values("Season Bench Pts", ascending=True)
        .set_index("playerName")[["Season Bench Pts"]]
    )
    st.bar_chart(bench_chart_df, horizontal=True, height=340)

with tab_matrices:
    ordered_managers = standings["playerName"].tolist()

    pts_mat = filtered_df.pivot(
        index="playerName", columns="Gameweek", values="actualGameweekPoints"
    ).reindex(ordered_managers)

    gw_rank_mat = (
        pts_mat.rank(axis=0, ascending=False, method="min").astype("Int64")
    )

    pos_mat = (
        filtered_df.pivot(index="playerName", columns="Gameweek", values="total_points")
        .reindex(ordered_managers)
        .rank(axis=0, ascending=False, method="min")
        .astype("Int64")
    )

    st.subheader("1. Gameweek Points Heatmap")
    st.dataframe(
        pts_mat.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.0f}", na_rep="-"),
        use_container_width=True,
    )

    st.subheader("2. Individual Gameweek Rankings (1 = Best in GW)")
    st.dataframe(
        gw_rank_mat.style.background_gradient(cmap="RdYlGn_r", axis=None, vmin=1, vmax=len(ordered_managers)),
        use_container_width=True,
    )

    st.subheader("3. Cumulative League Position Over Time (1 = 1st)")
    st.dataframe(
        pos_mat.style.background_gradient(cmap="RdYlGn_r", axis=None, vmin=1, vmax=len(ordered_managers)),
        use_container_width=True,
    )

with tab_raw:
    st.subheader("Filtered Gameweek Records")
    st.dataframe(
        filtered_df.sort_values(["Season", "Gameweek", "total_points"], ascending=[True, True, False]),
        hide_index=True,
        use_container_width=True,
    )