"""Audit four-hour monitoring records and compare station screening rules."""

import argparse
from calendar import isleap
from itertools import groupby
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
YEARS = list(range(2021, 2026))
THRESHOLDS = (70, 80, 90)
SLOT_HOURS = 4
HISTORY_STEPS = 30
HORIZONS = (4, 12, 24, 48, 72)
OBSERVATION_KEY = ["station_id", "observed_at"]
INDICATORS = ["do", "nh3n", "ph", "codmn"]
COUNT_COLUMNS = ["joint_complete_slots", "theoretical_4h_slots"]
COLUMNS = ["station_id", "year", *COUNT_COLUMNS]


def summarize_raw_observations(input_path):
    raw = pd.read_csv(input_path, dtype={"station_id": "string"})
    required = [*OBSERVATION_KEY, "source_snapshot_time", *INDICATORS]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {', '.join(sorted(missing))}")

    raw = raw.loc[:, required].copy()
    raw["station_id"] = raw["station_id"].str.strip()
    if raw.empty or raw["station_id"].isna().any() or raw["station_id"].eq("").any():
        raise ValueError("Provide at least one observation and a station_id for every row.")

    for column in ("observed_at", "source_snapshot_time"):
        timestamps = pd.to_datetime(raw[column], format="%Y-%m-%d %H:%M", errors="coerce")
        if timestamps.isna().any():
            raise ValueError(f"{column} must use YYYY-MM-DD HH:MM in local time.")
        raw[column] = timestamps

    times = raw["observed_at"]
    if not times.dt.year.isin(YEARS).all() or times.ne(times.dt.floor(f"{SLOT_HOURS}h")).any():
        raise ValueError("Observation timestamps must fall on the 4-hour grid in 2021–2025.")
    if raw["source_snapshot_time"].lt(times).any():
        raise ValueError("Source snapshots cannot precede their observations.")

    for indicator in INDICATORS:
        values = pd.to_numeric(raw[indicator], errors="coerce")
        raw[indicator] = values.where(values.abs().lt(float("inf")))

    raw["year"] = times.dt.year
    latest_time = raw.groupby(OBSERVATION_KEY)["source_snapshot_time"].transform("max")
    latest = raw.loc[raw["source_snapshot_time"].eq(latest_time)].drop_duplicates()
    if latest.duplicated(OBSERVATION_KEY).any():
        raise ValueError("Conflicting indicator values share the latest snapshot for an observation.")

    yearly_key = ["station_id", "year"]
    expected_index = pd.MultiIndex.from_product(
        [sorted(raw["station_id"].unique()), YEARS], names=yearly_key
    )
    annual = raw.groupby(yearly_key).size().reindex(expected_index, fill_value=0)
    audit = annual.to_frame("raw_rows")

    by_year = latest.groupby(yearly_key)
    observed = by_year.size().reindex(expected_index, fill_value=0)
    complete = latest[INDICATORS].notna().all(axis=1)
    complete_counts = complete.groupby([latest[key] for key in yearly_key]).sum()
    audit["observed_slots"] = observed
    audit["duplicate_rows_removed"] = audit["raw_rows"] - observed
    audit["joint_complete_slots"] = complete_counts.reindex(expected_index, fill_value=0)
    audit["incomplete_observed_slots"] = observed - audit["joint_complete_slots"]
    audit = audit.reset_index()
    audit["theoretical_4h_slots"] = audit["year"].map(
        {year: (365 + isleap(year)) * (24 // SLOT_HOURS) for year in YEARS}
    )
    audit["missing_slots"] = audit["theoretical_4h_slots"] - audit["observed_slots"]
    return audit, latest


def longest_gap_hours(available):
    longest = max((sum(1 for _ in run) for value, run in groupby(available) if not value), default=0)
    return longest * SLOT_HOURS


def summarize_windows(observations, station_years):
    groups = {key: group for key, group in observations.groupby(["station_id", "year"])}
    rows = []
    for row in station_years.itertuples(index=False):
        grid = pd.date_range(
            start=f"{row.year}-01-01", periods=row.theoretical_4h_slots, freq=f"{SLOT_HOURS}h"
        )
        station_year = groups.get((row.station_id, row.year))
        if station_year is None:
            complete = pd.Series(False, index=grid)
            missing_gap = incomplete_gap = None
        else:
            observed_times = station_year["observed_at"]
            observed = pd.Series(grid.isin(observed_times), index=grid)
            complete = station_year.set_index("observed_at")[INDICATORS].notna().all(axis=1)
            complete = complete.reindex(grid, fill_value=False)
            first, last = observed_times.min(), observed_times.max()
            missing_gap = longest_gap_hours(observed.loc[first:last])
            incomplete_gap = longest_gap_hours(complete.loc[first:last])

        usable_history = complete.rolling(HISTORY_STEPS).sum().eq(HISTORY_STEPS)
        for horizon in HORIZONS:
            steps_ahead = horizon // SLOT_HOURS
            usable_target = complete.shift(-steps_ahead, fill_value=False)
            usable_windows = int((usable_history & usable_target).sum())
            possible_windows = len(grid) - HISTORY_STEPS - steps_ahead + 1
            rows.append({
                "station_id": row.station_id,
                "year": row.year,
                "joint_coverage_pct": 100 * row.joint_complete_slots / len(grid),
                "longest_missing_record_gap_hours": missing_gap,
                "longest_incomplete_data_gap_hours": incomplete_gap,
                "forecast_horizon_hours": horizon,
                "possible_windows": possible_windows,
                "usable_windows": usable_windows,
                "usable_window_pct": 100 * usable_windows / possible_windows,
            })
    return pd.DataFrame(rows)


def annual_coverage(data):
    missing = set(COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    data = data.loc[:, COLUMNS].copy()
    data["station_id"] = data["station_id"].str.strip()
    if data.empty or data["station_id"].isna().any() or data["station_id"].eq("").any():
        raise ValueError("Provide at least one row and a station_id for each row.")

    for column in ["year", *COUNT_COLUMNS]:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any() or values.mod(1).ne(0).any():
            raise ValueError(f"{column} must contain whole numbers.")
        data[column] = values

    if not data["year"].isin(YEARS).all():
        raise ValueError("Expected years 2021 through 2025.")
    if data.duplicated(["station_id", "year"]).any():
        raise ValueError("Provide exactly one row per station and year.")

    complete = data["joint_complete_slots"]
    total = data["theoretical_4h_slots"]
    if ((total <= 0) | (complete < 0) | (complete > total)).any():
        raise ValueError("Expected 0 <= complete slots <= total slots, with total slots > 0.")

    data["coverage_pct"] = 100 * complete / total
    annual = data.pivot(index="station_id", columns="year", values="coverage_pct")
    annual = annual.reindex(columns=YEARS)
    if annual.isna().any().any():
        raise ValueError("Provide all five years for each station, using zero complete slots when appropriate.")
    return annual


def screen_stations(annual):
    stations = annual.rename(columns={year: f"coverage_{year}_pct" for year in YEARS})
    stations["mean_coverage_pct"] = annual.mean(axis=1)
    stations["min_coverage_pct"] = annual.min(axis=1)

    summary = []
    for threshold in THRESHOLDS:
        # Ignore floating-point noise when a coverage value equals the threshold.
        mean_pass = stations["mean_coverage_pct"] >= threshold - 1e-10
        every_year_pass = stations["min_coverage_pct"] >= threshold - 1e-10
        mean_only = mean_pass & ~every_year_pass
        stations[f"every_year_pass_{threshold}"] = every_year_pass
        stations[f"mean_only_pass_{threshold}"] = mean_only
        summary.append({
            "threshold_pct": threshold,
            "mean_pass": int(mean_pass.sum()),
            "every_year_pass": int(every_year_pass.sum()),
            "mean_only_pass": int(mean_only.sum()),
        })

    return stations.reset_index(), pd.DataFrame(summary)


def save_chart(summary, path):
    fig, ax = plt.subplots(figsize=(7.4, 4.2), layout="constrained")
    plot_data = summary.set_index("threshold_pct")[["mean_pass", "every_year_pass"]]
    plot_data.columns = ["Five-year mean", "Every year"]
    plot_data.plot.bar(ax=ax, color=["#82aaca", "#174d68"], width=0.66, rot=0)
    for bars in ax.containers:
        ax.bar_label(bars, padding=3)
    ax.set_title("How the coverage rule changes station selection", loc="left")
    ax.set_xlabel("Joint coverage threshold (%)")
    ax.set_ylabel("Stations passing")
    ax.set_ylim(0, max(1, float(plot_data.max().max())) * 1.25)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def save_window_chart(windows, path):
    overview = windows[windows["forecast_horizon_hours"].eq(HORIZONS[0])]
    fig, ax = plt.subplots(figsize=(7.4, 4.4), layout="constrained")
    gaps = overview["longest_missing_record_gap_hours"]
    has_gap_data = gaps.notna()
    points = ax.scatter(
        overview.loc[has_gap_data, "joint_coverage_pct"],
        overview.loc[has_gap_data, "usable_window_pct"],
        c=gaps.loc[has_gap_data] / 24,
        vmin=0,
        cmap="cividis",
        s=46,
        edgecolors="#314151",
        linewidths=0.4,
    )
    if has_gap_data.any():
        fig.colorbar(points, ax=ax, label="Longest internal outage (days)")
    if (~has_gap_data).any():
        ax.scatter(
            overview.loc[~has_gap_data, "joint_coverage_pct"],
            overview.loc[~has_gap_data, "usable_window_pct"],
            marker="x", color="#4d5660", label="No observations",
        )
        ax.legend(frameon=False)

    ax.set_title("Annual coverage and usable 4-hour forecast windows", loc="left")
    ax.set_xlabel("Annual joint coverage (%)")
    ax.set_ylabel("Usable windows (%)")
    ax.grid(alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", type=Path, help="Use a station-year totals CSV.")
    inputs.add_argument("--raw-input", type=Path, help="Use four-hour observation records.")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    quality = None
    windows = None
    try:
        if args.input is None:
            raw_input = args.raw_input or ROOT / "sample_raw_observations.csv"
            quality, observations = summarize_raw_observations(raw_input)
            windows = summarize_windows(observations, quality)
            annual_counts = quality
        else:
            annual_counts = pd.read_csv(args.input, dtype={"station_id": "string"})
        stations, summary = screen_stations(annual_coverage(annual_counts))
    except (OSError, ValueError) as error:
        parser.error(str(error))

    args.output.mkdir(parents=True, exist_ok=True)
    if quality is not None:
        quality.to_csv(args.output / "annual_data_quality.csv", index=False)
        windows.to_csv(args.output / "forecast_window_diagnostics.csv", index=False, float_format="%.4f")
        save_window_chart(windows, args.output / "coverage_vs_windows.png")
    stations.to_csv(args.output / "station_coverage.csv", index=False, float_format="%.4f")
    summary.to_csv(args.output / "threshold_summary.csv", index=False)
    save_chart(summary, args.output / "threshold_comparison.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
