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
    df.columns = [c.strip() for c in df.columns]
    df["Season"] = df["Season"].astype(str).str.strip()
    df["Gameweek"] = pd.to_numeric(df["Gameweek"], errors="coerce")
    df = df.dropna(subset=["Season", "Gameweek", "playerName"]).copy()
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
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def calc_table_height(num_rows: int) -> int:
    """Return exact pixel height so all rows fit without vertical scrolling."""
    return int((num_rows + 1) * 35 + 12)


try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find `{DATA_PATH}`. Add your CSV there and refresh.")
    st.stop()

if df.empty:
    st.warning("Your dataset is empty.")
    st.stop()

seasons = sorted(df["Season"].unique())

# --- Sidebar Filters (including Season dropdown) ---
st.sidebar.header("Dashboard Filters")

selected_season = st.sidebar.selectbox(
    "Season",
    options=seasons,
    index=len(seasons) - 1,
    key="season_selector",
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
        (standings["playerName"].map(prev_pos) - standings["Pos"]).fillna(0).astype(int)
    )
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
second_gap = (
    int(standings.iloc[1]["Gap to Above"])
    if len(standings) > 1 and pd.notna(standings.iloc[1]["Gap to Above"])
    else 0
)
top_bench = standings.sort_values("Season Bench Pts", ascending=False).iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("League Leader", f"{leader['playerName']}", f"{int(leader['total_points'])} pts")
k2.metric(
    "#1 vs #2 Gap",
    f"{second_gap} pts",
    standings.iloc[1]["playerName"] if len(standings) > 1 else "",
)
k3.metric("Avg GW Score", f"{filtered_df['actualGameweekPoints'].mean():.1f} pts")
k4.metric(
    "Most Bench Pts",
    f"{top_bench['playerName']}",
    f"{int(top_bench['Season Bench Pts'])} pts",
)

st.divider()

# --- Tabs for Visualisations & Matrices ---
tab_overview, tab_matrices, tab_alltime, tab_raw = st.tabs(
    [
        "📈 Standings & Charts",
        "🔥 GW & Position Matrices",
        "🏆 All-Time Seasons",
        "📄 Data Table",
    ]
)

with tab_overview:
    col_left, col_right = st.columns([1.25, 1.0])

    with col_left:
        st.subheader("Cumulative Points Trajectory")
        ordered_names = standings["playerName"].tolist()
        traj_long = filtered_df[["Gameweek", "playerName", "total_points"]].copy()
        traj_long["playerName"] = pd.Categorical(
            traj_long["playerName"], categories=ordered_names, ordered=True
        )
        traj_long = traj_long.sort_values(["playerName", "Gameweek"])

        vega_spec = {
            "mark": {"type": "line", "point": True},
            "encoding": {
                "x": {
                    "field": "Gameweek",
                    "type": "ordinal",
                    "title": "Gameweek",
                    "axis": {"labelAngle": 0},
                },
                "y": {
                    "field": "total_points",
                    "type": "quantitative",
                    "title": "Total Points",
                },
                "color": {
                    "field": "playerName",
                    "type": "nominal",
                    "title": "Manager",
                    "sort": ordered_names,
                    "legend": {
                        "orient": "bottom",
                        "columns": 4,
                        "labelLimit": 160,
                        "symbolLimit": 50,
                    },
                },
                "tooltip": [
                    {"field": "playerName", "type": "nominal", "title": "Manager"},
                    {"field": "Gameweek", "type": "ordinal", "title": "GW"},
                    {"field": "total_points", "type": "quantitative", "title": "Total Pts"},
                ],
            },
            "height": 460,
        }
        st.vega_lite_chart(traj_long, vega_spec, use_container_width=True)

    with col_right:
        st.subheader(f"GW {latest_gw} Standings")
        gw_pts_col = f"GW {latest_gw} Pts"
        standings_display = standings[
            [
                "Pos",
                "playerName",
                "total_points",
                "actualGameweekPoints",
                "Gap to Above",
                "Change in position",
            ]
        ].rename(
            columns={
                "playerName": "Manager",
                "total_points": "Total Pts",
                "actualGameweekPoints": gw_pts_col,
                "Change in position": "Δ Pos",
            }
        )

        max_abs_delta = max(1, int(standings_display["Δ Pos"].abs().max()))
        standings_styled = (
            standings_display.style
            .background_gradient(subset=["Pos"], cmap="RdYlGn_r", axis=0)
            .background_gradient(subset=["Total Pts", gw_pts_col], cmap="RdYlGn", axis=0)
            .background_gradient(subset=["Gap to Above"], cmap="Blues", axis=0)
            .background_gradient(
                subset=["Δ Pos"],
                cmap="RdYlGn",
                axis=0,
                vmin=-max_abs_delta,
                vmax=max_abs_delta,
            )
            .format(
                {
                    "Pos": "{:.0f}",
                    "Total Pts": "{:.0f}",
                    gw_pts_col: "{:.0f}",
                    "Gap to Above": "{:.0f}",
                    "Δ Pos": "{:+d}",
                },
                na_rep="-",
            )
        )

        st.dataframe(
            standings_styled,
            hide_index=True,
            use_container_width=True,
            height=calc_table_height(len(standings_display)),
        )

    st.divider()

    # --- Recent Form & Consistency ---
    st.subheader("Recent Form & Consistency")
    last3_cutoff = max(gw_min, latest_gw - 2)
    last5_cutoff = max(gw_min, latest_gw - 4)

    season_avg = (
        filtered_df.groupby("playerName")["actualGameweekPoints"]
        .mean()
        .rename("Season Avg")
    )
    last5_avg = (
        filtered_df[filtered_df["Gameweek"] >= last5_cutoff]
        .groupby("playerName")["actualGameweekPoints"]
        .mean()
        .rename(f"Last {latest_gw - last5_cutoff + 1} GWs Avg")
    )
    last3_avg = (
        filtered_df[filtered_df["Gameweek"] >= last3_cutoff]
        .groupby("playerName")["actualGameweekPoints"]
        .mean()
        .rename(f"Last {latest_gw - last3_cutoff + 1} GWs Avg")
    )
    std_dev = (
        filtered_df.groupby("playerName")["actualGameweekPoints"]
        .std()
        .fillna(0)
        .rename("Volatility (Std Dev)")
    )

    form_df = (
        pd.concat([season_avg, last5_avg, last3_avg, std_dev], axis=1)
        .reset_index()
        .rename(columns={"playerName": "Manager"})
    )
    last5_col = last5_avg.name
    form_df["Form vs Avg"] = form_df[last5_col] - form_df["Season Avg"]
    form_df = form_df.sort_values(last5_col, ascending=False).reset_index(drop=True)

    for c in ["Season Avg", last5_col, last3_avg.name, "Volatility (Std Dev)", "Form vs Avg"]:
        form_df[c] = form_df[c].round(1)

    form_left, form_right = st.columns([1.15, 1.0])
    with form_left:
        st.caption(f"Comparing {last5_col} against Season Avg")
        form_chart_long = form_df.melt(
            id_vars=["Manager"],
            value_vars=["Season Avg", last5_col],
            var_name="Metric",
            value_name="Avg Points",
        )
        form_vega = {
            "mark": "bar",
            "encoding": {
                "y": {
                    "field": "Manager",
                    "type": "nominal",
                    "sort": form_df["Manager"].tolist(),
                    "title": None,
                },
                "yOffset": {"field": "Metric", "type": "nominal"},
                "x": {
                    "field": "Avg Points",
                    "type": "quantitative",
                    "title": "Average GW Points",
                },
                "color": {
                    "field": "Metric",
                    "type": "nominal",
                    "legend": {"orient": "top", "title": None},
                },
                "tooltip": [
                    {"field": "Manager", "type": "nominal"},
                    {"field": "Metric", "type": "nominal"},
                    {"field": "Avg Points", "type": "quantitative"},
                ],
            },
            "height": max(380, len(form_df) * 28),
        }
        st.vega_lite_chart(form_chart_long, form_vega, use_container_width=True)

    with form_right:
        st.caption("Form & Consistency Breakdown")
        st.dataframe(
            form_df,
            hide_index=True,
            use_container_width=True,
            height=calc_table_height(len(form_df)),
        )

with tab_matrices:
    ordered_managers = standings["playerName"].tolist()
    num_managers = len(ordered_managers)
    full_height = calc_table_height(num_managers)

    pts_mat = filtered_df.pivot(
        index="playerName", columns="Gameweek", values="actualGameweekPoints"
    ).reindex(ordered_managers)
    pts_mat.index.name = "Manager"
    pts_mat.columns = [str(c) for c in pts_mat.columns]

    gw_rank_mat = pts_mat.rank(axis=0, ascending=False, method="min").astype("Int64")
    gw_rank_mat.index.name = "Manager"

    pos_mat = (
        filtered_df.pivot(index="playerName", columns="Gameweek", values="total_points")
        .reindex(ordered_managers)
        .rank(axis=0, ascending=False, method="min")
        .astype("Int64")
    )
    pos_mat.index.name = "Manager"
    pos_mat.columns = [str(c) for c in pos_mat.columns]

    compact_col_config = {
        "Manager": st.column_config.TextColumn("Manager", width="medium", pinned=True),
        **{col: st.column_config.NumberColumn(col, width=42) for col in pts_mat.columns},
    }

    st.subheader("1. Gameweek Points Heatmap (Shaded Relative to Each Gameweek)")
    st.dataframe(
        pts_mat.style.background_gradient(cmap="RdYlGn", axis=0).format(
            "{:.0f}", na_rep="-"
        ),
        use_container_width=True,
        height=full_height,
        column_config=compact_col_config,
    )

    st.markdown("---")

    st.subheader("2. Individual Gameweek Rankings (1 = Best in GW)")
    st.dataframe(
        gw_rank_mat.style.background_gradient(
            cmap="RdYlGn_r", axis=0, vmin=1, vmax=max(1, num_managers)
        ),
        use_container_width=True,
        height=full_height,
        column_config=compact_col_config,
    )

    st.markdown("---")

    st.subheader("3. Cumulative League Position Over Time (1 = 1st)")
    st.dataframe(
        pos_mat.style.background_gradient(
            cmap="RdYlGn_r", axis=0, vmin=1, vmax=max(1, num_managers)
        ),
        use_container_width=True,
        height=full_height,
        column_config=compact_col_config,
    )

with tab_alltime:
    st.subheader("Multi-Season Summary (All Available Seasons)")
    final_gws = df.groupby(["Season", "playerName"], as_index=False)["Gameweek"].max()
    season_finals = df.merge(final_gws, on=["Season", "playerName", "Gameweek"])
    season_finals["Season Pos"] = (
        season_finals.groupby("Season")["total_points"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    all_time_summary = (
        df.groupby("playerName", as_index=False)
        .agg(
            Seasons_Played=("Season", "nunique"),
            Total_GWs=("Gameweek", "count"),
            Avg_GW_Pts=("actualGameweekPoints", "mean"),
            Best_GW=("actualGameweekPoints", "max"),
            Total_Bench_Pts=("points_on_bench", "sum"),
        )
        .merge(
            season_finals.groupby("playerName", as_index=False).agg(
                Titles=("Season Pos", lambda s: int((s == 1).sum())),
                Podiums=("Season Pos", lambda s: int((s <= 3).sum())),
                Best_Finish=("Season Pos", "min"),
            ),
            on="playerName",
            how="left",
        )
        .sort_values(["Titles", "Podiums", "Avg_GW_Pts"], ascending=[False, False, False])
    )
    all_time_summary["Avg_GW_Pts"] = all_time_summary["Avg_GW_Pts"].round(1)

    st.dataframe(
        all_time_summary.rename(
            columns={
                "playerName": "Manager",
                "Seasons_Played": "Seasons",
                "Total_GWs": "GWs",
                "Avg_GW_Pts": "Avg GW Pts",
                "Best_GW": "Best GW",
                "Total_Bench_Pts": "Total Bench Pts",
                "Best_Finish": "Best Finish",
            }
        ),
        hide_index=True,
        use_container_width=True,
        height=calc_table_height(len(all_time_summary)),
    )

    st.subheader("Final / Latest Points by Season")
    season_comp = season_finals.pivot(
        index="playerName", columns="Season", values="total_points"
    )
    st.dataframe(
        season_comp.style.format("{:.0f}", na_rep="-"),
        use_container_width=True,
        height=calc_table_height(len(season_comp)),
    )

with tab_raw:
    st.subheader("Filtered Gameweek Records")
    st.dataframe(
        filtered_df.sort_values(
            ["Season", "Gameweek", "total_points"], ascending=[True, True, False]
        ),
        hide_index=True,
        use_container_width=True,
    )
