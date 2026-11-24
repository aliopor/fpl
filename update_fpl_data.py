from pathlib import Path
import time
import pandas as pd
import requests

LEAGUE_ID = 752891
SEASON = "2026-2027"
OUTPUT_CSV = Path("data/fpl_history.csv")
OVERWRITE_ONLY_ACTIVE_SEASON = True

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FPL-League-Tracker/1.0)"
}


def fetch_json(url: str, session: requests.Session) -> dict:
    response = session.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return response.json()


def get_league_entries(league_id: int, session: requests.Session) -> pd.DataFrame:
    entries = []
    page = 1

    while True:
        url = f"https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/?page_standings={page}"
        data = fetch_json(url, session)
        results = data.get("standings", {}).get("results", [])
        entries.extend(results)

        if not data.get("standings", {}).get("has_next"):
            break
        page += 1

    if not entries:
        raise ValueError(f"No entries found for league {league_id}.")

    return pd.DataFrame(entries)


def download_season_scores(league_id: int, season: str) -> pd.DataFrame:
    all_rows = []

    with requests.Session() as session:
        league_df = get_league_entries(league_id, session)

        for entry_id, player_name in zip(league_df["entry"], league_df["player_name"]):
            history_url = f"https://fantasy.premierleague.com/api/entry/{int(entry_id)}/history/"
            history_data = fetch_json(history_url, session)

            current_df = pd.DataFrame(history_data.get("current", []))
            if current_df.empty:
                continue

            current_df["playerID"] = int(entry_id)
            current_df["playerName"] = player_name
            all_rows.append(current_df)
            time.sleep(0.15)

    if not all_rows:
        raise ValueError("No gameweek history rows were returned.")

    scores_df = pd.concat(all_rows, ignore_index=True)

    if "event" in scores_df.columns and "Gameweek" not in scores_df.columns:
        scores_df = scores_df.rename(columns={"event": "Gameweek"})

    scores_df["actualGameweekPoints"] = (
        scores_df["points"] - scores_df["event_transfers_cost"]
    )
    scores_df["Season"] = season

    preferred_order = [
        "Gameweek",
        "points",
        "total_points",
        "rank",
        "rank_sort",
        "overall_rank",
        "bank",
        "value",
        "event_transfers",
        "event_transfers_cost",
        "points_on_bench",
        "playerID",
        "playerName",
        "actualGameweekPoints",
        "Season",
    ]
    ordered_cols = [c for c in preferred_order if c in scores_df.columns] + [
        c for c in scores_df.columns if c not in preferred_order
    ]
    scores_df = (
        scores_df[ordered_cols]
        .sort_values(["Gameweek", "total_points"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return scores_df


def save_scores(new_season_df: pd.DataFrame, output_path: Path, overwrite_only_season: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite_only_season and output_path.exists():
        existing_df = pd.read_csv(output_path)
        if "Season" in existing_df.columns:
            kept_df = existing_df[existing_df["Season"].astype(str) != SEASON]
            final_df = pd.concat([kept_df, new_season_df], ignore_index=True)
        else:
            final_df = new_season_df
    else:
        final_df = new_season_df

    final_df.to_csv(output_path, index=False)
    print(f"Saved {len(new_season_df)} rows for {SEASON} to {output_path} (total rows: {len(final_df)}).")


if __name__ == "__main__":
    season_df = download_season_scores(LEAGUE_ID, SEASON)
    save_scores(season_df, OUTPUT_CSV, OVERWRITE_ONLY_ACTIVE_SEASON)
