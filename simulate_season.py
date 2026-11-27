from pathlib import Path
import numpy as np
import pandas as pd


def run_fpl_bayessim_v3(
    input_file: str = "data/fpl_history.csv",
    current_season: str = "2026-2027",
    total_gameweeks: int = 38,
    k_inertia: float = 38.0,
    min_hist_gws: int = 10,
    n_sims: int = 50000,
    seed: int = 42,
    output_csv: str = "data/predictions.csv",
) -> pd.DataFrame:
    """Runs FPL-BayesSim-v3 and appends/overwrites (Season, Gameweek) in predictions format."""
    file_path = Path(input_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find input file: {input_file}")

    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(file_path, sheet_name=xls.sheet_names[0])
    elif file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Input file must be an Excel (.xlsx/.xls) or CSV (.csv) file.")

    df.columns = [c.strip() for c in df.columns]
    if "actualGameweekPoints" in df.columns:
        df = df.rename(columns={"actualGameweekPoints": "Points"})
    elif "Points" not in df.columns:
        raise ValueError("Expected column 'actualGameweekPoints' or 'Points' in dataset.")

    df["Season"] = df["Season"].astype(str).str.strip()
    df["Gameweek"] = pd.to_numeric(df["Gameweek"], errors="coerce")
    df = df.dropna(subset=["Season", "Gameweek", "playerName", "Points"]).copy()
    df["Gameweek"] = df["Gameweek"].astype(int)
    df["Points"] = pd.to_numeric(df["Points"], errors="coerce")

    hist_df = df[df["Season"] != current_season].copy()
    cur_df = df[df["Season"] == current_season].copy()

    if cur_df.empty:
        raise ValueError(f"No records found for current season '{current_season}'.")

    played_gws = sorted(cur_df["Gameweek"].unique())
    latest_gw = int(max(played_gws))
    n_played = len(played_gws)
    n_remaining = total_gameweeks - n_played

    if n_remaining < 0:
        raise ValueError(
            f"Played gameweeks ({n_played}) exceed total gameweeks ({total_gameweeks})."
        )

    # Historical league prior & empirical residuals (score minus manager's own mean)
    if not hist_df.empty:
        grand_mean = float(hist_df["Points"].mean())
        league_residuals = (
            hist_df["Points"]
            - hist_df.groupby("playerName")["Points"].transform("mean")
        ).dropna().values
    else:
        grand_mean = float(cur_df["Points"].mean())
        league_residuals = (
            cur_df["Points"]
            - cur_df.groupby("playerName")["Points"].transform("mean")
        ).dropna().values

    if len(league_residuals) == 0:
        league_residuals = np.array([0.0])

    # Current season standings
    standings = cur_df.pivot_table(
        index="playerName", columns="Gameweek", values="Points", aggfunc="first"
    )
    gw_cols = list(standings.columns)
    standings["Total_Locked"] = standings[gw_cols].sum(axis=1)
    standings["Avg_Pts"] = standings[gw_cols].mean(axis=1)
    standings = standings.sort_values(by="Total_Locked", ascending=False)

    managers = standings.index.tolist()
    n_managers = len(managers)
    locked_pts_dict = standings["Total_Locked"].to_dict()
    gw_avg_dict = standings["Avg_Pts"].to_dict()

    w_n = n_played / (n_played + k_inertia)

    rng = np.random.default_rng(seed)
    sim_totals = np.zeros((n_sims, n_managers), dtype=float)

    for idx, manager in enumerate(managers):
        locked_pts = locked_pts_dict[manager]
        if n_remaining == 0:
            sim_totals[:, idx] = locked_pts
            continue

        m_hist = (
            hist_df[hist_df["playerName"] == manager]["Points"].dropna().values
            if not hist_df.empty
            else np.array([])
        )

        if len(m_hist) >= min_hist_gws:
            samples = rng.choice(m_hist, size=(n_sims, n_remaining), replace=True)
            sim_totals[:, idx] = locked_pts + samples.sum(axis=1)
        else:
            x_bar = gw_avg_dict[manager]
            mu_post = (1.0 - w_n) * grand_mean + w_n * x_bar
            resids = rng.choice(league_residuals, size=(n_sims, n_remaining), replace=True)
            weekly_scores = np.clip(np.round(mu_post + resids), 0, None)
            sim_totals[:, idx] = locked_pts + weekly_scores.sum(axis=1)

    # Rank 1 = highest simulated points, n_managers = lowest
    order_indices = np.argsort(sim_totals, axis=1)
    ranks = np.empty_like(order_indices)
    rows = np.arange(n_sims)[:, None]
    ranks[rows, order_indices] = np.arange(n_managers, 0, -1)

    # Order managers by expected finish position (1st-most-likely first)
    mean_ranks = ranks.mean(axis=0)
    sorted_idx = np.argsort(mean_ranks)
    sorted_managers = [managers[i] for i in sorted_idx]

    sim_rows = []
    for manager in sorted_managers:
        m_idx = managers.index(manager)
        m_ranks = ranks[:, m_idx]
        for pos in range(1, n_managers + 1):
            odds = round(float((m_ranks == pos).mean()), 4)
            sim_rows.append(
                {
                    "Player": manager,
                    "Gameweek": latest_gw,
                    "Position": pos,
                    "Odds": odds,
                    "Season": current_season,
                }
            )

    gw_predictions_df = pd.DataFrame(
        sim_rows, columns=["Player", "Gameweek", "Position", "Odds", "Season"]
    )

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        existing_df["Season"] = existing_df["Season"].astype(str).str.strip()
        existing_df["Gameweek"] = pd.to_numeric(existing_df["Gameweek"], errors="coerce").astype(int)
        mask_same = (existing_df["Season"] == current_season) & (
            existing_df["Gameweek"] == latest_gw
        )
        kept_df = existing_df[~mask_same]
        final_df = pd.concat([kept_df, gw_predictions_df], ignore_index=True)
    else:
        final_df = gw_predictions_df

    final_df = final_df.sort_values(
        ["Season", "Gameweek", "Player", "Position"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    final_df.to_csv(out_path, index=False)

    print(
        f"Simulated {current_season} after GW{latest_gw} ({n_remaining} GWs left) -> "
        f"saved {len(gw_predictions_df)} rows ({len(final_df)} total in {out_path})."
    )
    return gw_predictions_df


if __name__ == "__main__":
    run_fpl_bayessim_v3()