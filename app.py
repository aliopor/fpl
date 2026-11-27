from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="FPL Windlab League Dashboard",
    page_icon="⚽",
    layout="wide",
)

DATA_PATH = "data/fpl_history.csv"
PREDICTIONS_PATH = "data/predictions.csv"


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


@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    pred = pd.read_csv(path)
    pred.columns = [c.strip() for c in pred.columns]
    pred["Season"] = pred["Season"].astype(str).str.strip()
    pred["Player"] = pred["Player"].astype(str).str.strip()
    pred["Gameweek"] = pd.to_numeric(pred["Gameweek"], errors="coerce")
    pred["Position"] = pd.to_numeric(pred["Position"], errors="coerce")
    pred["Odds"] = pd.to_numeric(pred["Odds"], errors="coerce")
    pred = pred.dropna(subset=["Player", "Gameweek", "Position", "Odds", "Season"]).copy()
    pred["Gameweek"] = pred["Gameweek"].astype(int)
    pred["Position"] = pred["Position"].astype(int)
    return pred


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

# --- Core Standings & Deltas Across Selected Gameweek Range ---
latest_gw = int(filtered_df["Gameweek"].max())
prev_gw = latest_gw - 1 if latest_gw > gw_min else None

standings = filtered_df[filtered_df["Gameweek"] == latest_gw].copy()

if gw_min == min_gw:
    standings["Range Pts"] = standings["total_points"]
    range_pts_label = "Total Pts"
else:
    range_sums = (
        filtered_df.groupby("playerName", as_index=False)["actualGameweekPoints"]
        .sum()
        .rename(columns={"actualGameweekPoints": "Range Pts"})
    )
    standings = standings.drop(columns=["Range Pts"], errors="ignore").merge(
        range_sums, on="playerName", how="left"
    )
    range_pts_label = f"GW {gw_min}–{gw_max} Pts"

standings = standings.sort_values("Range Pts", ascending=False).reset_index(drop=True)
standings["Pos"] = standings["Range Pts"].rank(ascending=False, method="min").astype(int)
standings["Gap to Above"] = standings["Range Pts"].shift(1) - standings["Range Pts"]

if prev_gw is not None and not filtered_df[filtered_df["Gameweek"] == prev_gw].empty:
    if gw_min == min_gw:
        prev_scores = (
            filtered_df[filtered_df["Gameweek"] == prev_gw]
            .set_index("playerName")["total_points"]
        )
    else:
        prev_scores = (
            filtered_df[filtered_df["Gameweek"] <= prev_gw]
            .groupby("playerName")["actualGameweekPoints"]
            .sum()
        )
    prev_pos = prev_scores.rank(ascending=False, method="min")
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
st.title(f"⚽ FPL Windlab League Dashboard — {selected_season}")
st.caption(f"Showing Gameweeks {gw_min}–{gw_max} (Latest: GW {latest_gw})")

leader = standings.iloc[0]
second_gap = (
    int(standings.iloc[1]["Gap to Above"])
    if len(standings) > 1 and pd.notna(standings.iloc[1]["Gap to Above"])
    else 0
)
top_bench = standings.sort_values("Season Bench Pts", ascending=False).iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("League Leader", f"{leader['playerName']}", f"{int(leader['Range Pts'])} pts")
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
tab_overview, tab_matrices, tab_sim, tab_alltime, tab_raw = st.tabs(
    [
        "📈 Standings & Charts",
        "🔥 GW & Position Matrices",
        "🎲 Season Predictions",
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
            "height": 600,
        }
        st.vega_lite_chart(traj_long, vega_spec, use_container_width=True)

    with col_right:
        standings_heading = (
            f"GW {latest_gw} Standings"
            if gw_min == min_gw
            else f"Standings (GW {gw_min}–{gw_max})"
        )
        st.subheader(standings_heading)
        gw_pts_col = f"GW {latest_gw} Pts"
        standings_display = standings[
            [
                "Pos",
                "playerName",
                "Range Pts",
                "actualGameweekPoints",
                "Gap to Above",
                "Change in position",
            ]
        ].rename(
            columns={
                "playerName": "Manager",
                "Range Pts": range_pts_label,
                "actualGameweekPoints": gw_pts_col,
                "Change in position": "Δ Pos",
            }
        )

        # Apply colour gradients only to range/total points and GW Pts columns
        standings_styled = (
            standings_display.style
            .background_gradient(
                subset=[range_pts_label, gw_pts_col], cmap="RdYlGn", axis=0
            )
            .format(
                {
                    "Pos": "{:.0f}",
                    range_pts_label: "{:.0f}",
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

with tab_sim:
    st.subheader(f"Monte Carlo Season Finish Probabilities — {selected_season}")
    pred_path = Path(PREDICTIONS_PATH)

    if pred_path.exists():
        pred_df = load_predictions(PREDICTIONS_PATH)
        season_preds = pred_df[pred_df["Season"] == selected_season].copy()

        if not season_preds.empty:
            avail_sim_gws = sorted(season_preds["Gameweek"].unique())
            default_gw_idx = (
                avail_sim_gws.index(gw_max)
                if gw_max in avail_sim_gws
                else len(avail_sim_gws) - 1
            )
            sim_gw = st.selectbox(
                "Simulation Snapshot (After Gameweek)",
                options=avail_sim_gws,
                index=default_gw_idx,
                key="sim_gw_selector",
            )

            gw_sim = season_preds[season_preds["Gameweek"] == sim_gw].copy()
            if selected_managers:
                gw_sim_filtered = gw_sim[gw_sim["Player"].isin(selected_managers)]
            else:
                gw_sim_filtered = gw_sim

            prob_pivot = (
                gw_sim_filtered.pivot(index="Player", columns="Position", values="Odds")
                .fillna(0.0)
            )
            positions = sorted(prob_pivot.columns)
            exp_pos = (prob_pivot[positions].values * np.array(positions)).sum(axis=1)

            order = (
                prob_pivot.assign(
                    _p1=prob_pivot.get(1, 0.0),
                    _podium=prob_pivot[[p for p in positions if p <= 3]].sum(axis=1),
                    _ep=exp_pos,
                )
                .sort_values(["_p1", "_podium", "_ep"], ascending=[False, False, True])
                .index
            )

            prob_pct = (prob_pivot.loc[order, positions] * 100).round(0).astype(int)
            prob_pct.index.name = "Manager"
            prob_pct.columns = [str(p) for p in prob_pct.columns]

            sim_col_config = {
                "Manager": st.column_config.TextColumn("Manager", width="medium", pinned=True),
                **{col: st.column_config.NumberColumn(col, width=42) for col in prob_pct.columns},
            }

            st.caption(
                f"End-of-season finish position probabilities (%) simulated after GW {sim_gw}."
            )
            st.dataframe(
                prob_pct.style.background_gradient(
                    cmap="RdYlGn_r",
                    axis=None,
                    vmin=1,
                    vmax=max(1, int(prob_pct.values.max())),
                ).format("{:d}"),
                use_container_width=True,
                height=calc_table_height(len(prob_pct)),
                column_config=sim_col_config,
            )

            st.markdown("---")
            st.subheader("Title Odds Trajectory Over the Season")
            title_traj = season_preds[season_preds["Position"] == 1].copy()
            if selected_managers:
                title_traj = title_traj[title_traj["Player"].isin(selected_managers)]
            title_traj["Win_Pct"] = (title_traj["Odds"] * 100).round(1)
            ordered_sim_names = list(order)

            title_vega = {
                "mark": {"type": "line", "point": True},
                "encoding": {
                    "x": {
                        "field": "Gameweek",
                        "type": "ordinal",
                        "title": "Gameweek",
                        "axis": {"labelAngle": 0},
                    },
                    "y": {
                        "field": "Win_Pct",
                        "type": "quantitative",
                        "title": "1st Place Probability (%)",
                    },
                    "color": {
                        "field": "Player",
                        "type": "nominal",
                        "title": "Manager",
                        "sort": ordered_sim_names,
                        "legend": {
                            "orient": "bottom",
                            "columns": 4,
                            "labelLimit": 160,
                            "symbolLimit": 50,
                        },
                    },
                    "tooltip": [
                        {"field": "Player", "type": "nominal", "title": "Manager"},
                        {"field": "Gameweek", "type": "ordinal", "title": "GW"},
                        {"field": "Win_Pct", "type": "quantitative", "title": "Win %"},
                    ],
                },
                "height": 440,
            }
            st.vega_lite_chart(title_traj, title_vega, use_container_width=True)
        else:
            st.info(f"No simulation predictions found for season {selected_season}.")
    else:
        st.info(
            f"Could not find `{PREDICTIONS_PATH}`. Run `simulate_season.py` to generate end-of-season predictions."
        )

with tab_alltime:
    st.subheader("Multi-Season Summary (All Available Seasons)")
    season_max_gws = (
        df.groupby("Season", as_index=False)["Gameweek"]
        .max()
        .rename(columns={"Gameweek": "Season_Max_GW"})
    )
    completed_seasons = set(
        season_max_gws[season_max_gws["Season_Max_GW"] >= 38]["Season"]
    )

    final_gws = df.groupby(["Season", "playerName"], as_index=False)["Gameweek"].max()
    season_finals = df.merge(final_gws, on=["Season", "playerName", "Gameweek"])
    season_finals["Season Pos"] = (
        season_finals.groupby("Season")["total_points"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    completed_finals = season_finals[season_finals["Season"].isin(completed_seasons)]
    if completed_finals.empty:
        honours_df = pd.DataFrame(
            columns=["playerName", "Titles", "Podiums", "Best_Finish"]
        )
    else:
        honours_df = completed_finals.groupby("playerName", as_index=False).agg(
            Titles=("Season Pos", lambda s: int((s == 1).sum())),
            Podiums=("Season Pos", lambda s: int((s <= 3).sum())),
            Best_Finish=("Season Pos", "min"),
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
        .merge(honours_df, on="playerName", how="left")
    )
    all_time_summary["Titles"] = all_time_summary["Titles"].fillna(0).astype(int)
    all_time_summary["Podiums"] = all_time_summary["Podiums"].fillna(0).astype(int)
    all_time_summary["Best_Finish"] = all_time_summary["Best_Finish"].astype("Int64")
    all_time_summary["Avg_GW_Pts"] = all_time_summary["Avg_GW_Pts"].round(1)
    all_time_summary = all_time_summary.sort_values(
        ["Titles", "Podiums", "Avg_GW_Pts"], ascending=[False, False, False]
    )

    if completed_seasons:
        st.caption(
            f"Titles, Podiums, and Best Finish count completed 38-GW seasons only ({', '.join(sorted(completed_seasons))})."
        )
    else:
        st.caption("No completed 38-GW seasons yet.")

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
