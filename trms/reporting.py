"""Write every artefact of a run to disk (CSV, JSON, NPZ, Excel, Markdown)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import global_metrics, step_metrics


def _json(obj, path: Path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    path.write_text(json.dumps(obj, indent=2, default=conv))


def controller_metrics(df, steps, Ts):
    t = df["time"].values
    srows = []
    for s in steps:
        y = df["pitch" if s["axis"] == "pitch" else "yaw"].values
        m = step_metrics(t, y, s)
        other = "yaw" if s["axis"] == "pitch" else "pitch"
        seg = (t >= s["t_start"] - 1e-9) & (t < s["t_end"] - 1e-9)
        dev = np.abs(df[other].values[seg] - df[f"ref_{other}"].values[seg])
        m["cross_coupling_axis"] = other
        m["cross_coupling_peak_dev_rad"] = float(dev.max()) if dev.size else np.nan
        m["cross_coupling_peak_dev_deg"] = float(np.degrees(m["cross_coupling_peak_dev_rad"]))
        srows.append(m)
    sdf = pd.DataFrame(srows)
    g = global_metrics(t, df[["ref_pitch", "ref_yaw"]].values, df[["pitch", "yaw"]].values,
                       df[["u_pitch", "u_yaw"]].values, Ts, df["compute_time_ms"].values)
    for ax in ("pitch", "yaw"):
        sub = sdf[sdf["axis"] == ax]
        g[f"mean_overshoot_pct_{ax}"] = float(sub["overshoot_pct"].mean())
        g[f"max_overshoot_pct_{ax}"] = float(sub["overshoot_pct"].max())
        g[f"mean_settling_time_s_{ax}"] = float(sub["settling_time_2pct_s"].mean())
        g[f"max_settling_time_s_{ax}"] = float(sub["settling_time_2pct_s"].max())
        g[f"steps_not_settled_{ax}"] = int((~sub["settled"]).sum())
        g[f"mean_rise_time_s_{ax}"] = float(sub["rise_time_10_90_s"].mean())
    return sdf, g


def write_controller_folder(folder: Path, name, df, sdf, g, describe, tuning_entry,
                            predictions=None):
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / "telemetry.csv", index=False)
    sdf.to_csv(folder / "step_metrics.csv", index=False)
    pd.Series(g, name=name).to_csv(folder / "global_metrics.csv", header=["value"])
    _json({"controller": describe, "tuning": tuning_entry}, folder / "controller_parameters.json")
    _json(g, folder / "global_metrics.json")
    if predictions:
        np.savez_compressed(
            folder / "nmpc_predictions.npz",
            t=np.array([p["t"] for p in predictions]),
            X=np.stack([p["X"] for p in predictions]),      # (K, N+1, 6) predicted states
            U=np.stack([p["U"] for p in predictions]),      # (K, N, 2) planned inputs
            R=np.stack([p["R"] for p in predictions]))      # (K, N, 2) references used
    desc = (folder / "README.txt")
    desc.write_text(
        f"{name} simulation outputs\n"
        "telemetry.csv            one row per control sample, every state, derived quantity and controller internal\n"
        "step_metrics.csv         overshoot, rise/settling time, steady-state error, IAE/ISE/ITAE for every reference step\n"
        "global_metrics.csv/json  whole-run error, effort, smoothness and compute-time statistics\n"
        "controller_parameters.json  final parameters + full tuning report (objective, history, bounds)\n"
        + ("nmpc_predictions.npz     predicted state/input trajectories at every sample (keys t, X, U, R)\n"
           if predictions else "")
        + "*.png                    figures\n")


def write_comparison(root: Path, dfs: dict, step_tables: dict, globals_: dict, params_doc: dict):
    names = list(dfs)
    # merged time series
    merged = None
    for n, df in dfs.items():
        cols = ["time", "ref_pitch", "ref_yaw", "pitch", "yaw", "err_pitch", "err_yaw",
                "u_pitch", "u_yaw", "cum_IAE_pitch", "cum_IAE_yaw", "cum_int_u2_pitch",
                "cum_int_u2_yaw", "compute_time_ms"]
        d = df[cols].copy()
        d = d.rename(columns={c: f"{c}_{n}" for c in cols if c not in ("time", "ref_pitch", "ref_yaw")})
        merged = d if merged is None else merged.merge(d.drop(columns=["ref_pitch", "ref_yaw"]), on="time")
    merged.to_csv(root / "comparison_timeseries.csv", index=False)

    step_df = pd.concat([sdf.assign(controller=n, step_id=np.arange(len(sdf)))
                         for n, sdf in step_tables.items()], ignore_index=True)
    lead = ["controller", "step_id", "axis", "t_start", "r_from", "r_to"]
    step_df = step_df[lead + [c for c in step_df.columns if c not in lead]]
    step_df.to_csv(root / "comparison_step_metrics.csv", index=False)

    glob_df = pd.DataFrame(globals_).T
    glob_df.index.name = "controller"
    glob_df.T.to_csv(root / "comparison_global_metrics.csv")

    # side-by-side step table: one row per step, metrics for each controller
    vals = ["overshoot_pct", "settling_time_2pct_s", "rise_time_10_90_s",
            "steady_state_error", "IAE", "cross_coupling_peak_dev_deg"]
    wide = step_df.pivot(index="step_id", columns="controller", values=vals)
    wide.columns = [f"{m}__{c}" for m, c in wide.columns]
    info = step_df[step_df["controller"] == names[0]].set_index("step_id")[["axis", "t_start", "r_from", "r_to"]]
    wide = info.join(wide).reset_index()

    with pd.ExcelWriter(root / "comparison.xlsx", engine="openpyxl") as xw:
        glob_df.T.to_excel(xw, sheet_name="global_metrics")
        wide.to_excel(xw, sheet_name="step_side_by_side", index=False)
        step_df.to_excel(xw, sheet_name="step_metrics_long", index=False)
        merged.to_excel(xw, sheet_name="timeseries", index=False)
        for n, df in dfs.items():
            df.to_excel(xw, sheet_name=f"telemetry_{n}"[:31], index=False)
        pd.json_normalize(params_doc, sep=".").T.to_excel(xw, sheet_name="parameters", header=["value"])
    return step_df, glob_df, wide


def write_summary(root: Path, glob_df, wide, params_doc, run_info):
    def fmt(v, nd=3):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return str(v)
        return "not settled" if not np.isfinite(v) else f"{v:.{nd}g}"

    names = list(glob_df.index)
    rows = [("IAE pitch [rad·s]", "IAE_pitch"), ("IAE yaw [rad·s]", "IAE_yaw"),
            ("RMSE pitch [rad]", "RMSE_pitch"), ("RMSE yaw [rad]", "RMSE_yaw"),
            ("Mean overshoot pitch [%]", "mean_overshoot_pct_pitch"),
            ("Mean overshoot yaw [%]", "mean_overshoot_pct_yaw"),
            ("Mean settling time pitch, settled steps only [s]", "mean_settling_time_s_pitch"),
            ("Mean settling time yaw, settled steps only [s]", "mean_settling_time_s_yaw"),
            ("Steps not settled (pitch / yaw)", None),
            ("Control effort ∫u² (total)", "effort_int_u2_total"),
            ("Input total variation (total)", "total_variation_u_total"),
            ("Saturated samples pitch [%]", "saturation_fraction_pitch"),
            ("Compute time mean / max [ms]", None)]
    lines = ["# TRMS reference tracking: PID vs N-MPC", "",
             f"Run: `{run_info['run_dir']}`  ", f"Created: {run_info['created']}  ",
             f"Evaluation scenario: {run_info['scenario']}", "",
             "## Whole-run metrics (lower is better)", "",
             "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for lab, k in rows:
        if k is None and lab.startswith("Steps"):
            vals = [f"{int(glob_df.loc[n, 'steps_not_settled_pitch'])} / "
                    f"{int(glob_df.loc[n, 'steps_not_settled_yaw'])}" for n in names]
        elif k is None:
            vals = [f"{glob_df.loc[n, 'compute_time_mean_ms']:.2f} / "
                    f"{glob_df.loc[n, 'compute_time_max_ms']:.1f}" for n in names]
        elif k == "saturation_fraction_pitch":
            vals = [f"{100 * glob_df.loc[n, k]:.1f}" for n in names]
        else:
            vals = [fmt(glob_df.loc[n, k]) for n in names]
        lines.append(f"| {lab} | " + " | ".join(vals) + " |")
    lines += ["", "## Per-step results", "",
              "| Step | " + " | ".join(f"OS% {n} | Ts2% {n} [s]" for n in names) + " |",
              "|---|" + "---|---|" * len(names)]
    for _, r in wide.iterrows():
        lab = f"{r['axis']} @{r['t_start']:.0f}s ({np.degrees(r['r_from']):.1f}°→{np.degrees(r['r_to']):.1f}°)"
        cells = []
        for n in names:
            cells += [fmt(r[f'overshoot_pct__{n}']), fmt(r[f'settling_time_2pct_s__{n}'])]
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    lines += ["", "## Controller parameters", "", "```json",
              __import__("json").dumps(params_doc["controllers"], indent=2), "```", ""]
    (root / "summary.md").write_text("\n".join(lines))
