"""Shared set-up for the N-MPC studies (horizon sweep, model-mismatch robustness).

Every study reuses the project's own settings, cached tuned parameters, evaluation
scenario, plant, controller and metrics, so its numbers are directly comparable
with the main run (``python main.py``). Nothing here re-tunes anything: the studies
change ONE thing at a time (the horizon, or the plant parameters) and keep every
other tuned value fixed.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from trms.controllers.nmpc import NMPCController
from trms.model import rest_state
from trms.params import TRMSParams
from trms.reporting import controller_metrics
from trms.scenario import Scenario
from trms.simulation import run_closed_loop

ROOT = Path(__file__).resolve().parents[1]

# columns reported by every study, in this order
REPORT_COLUMNS = [
    "diverged", "IAE_pitch", "IAE_yaw", "IAE_total", "RMSE_pitch", "RMSE_yaw",
    "max_overshoot_pct_pitch", "max_overshoot_pct_yaw",
    "steps_not_settled_pitch", "steps_not_settled_yaw",
    "mean_settling_time_s_pitch", "mean_settling_time_s_yaw",
    "effort_int_u2_total", "saturation_fraction_pitch",
    "compute_time_mean_ms", "compute_time_p95_ms", "compute_time_max_ms",
    "realtime_budget_exceeded_fraction",
]


def load_setup(settings_path: Path | None = None, cache_path: Path | None = None) -> dict:
    """Settings, nominal parameters and the cached (full-budget) N-MPC parameters."""
    cfg = yaml.safe_load(Path(settings_path or ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    cache = json.loads(Path(cache_path or ROOT / "config" / "tuned_parameters.json").read_text(encoding="utf-8"))
    if "nmpc" not in cache:
        raise SystemExit("No tuned N-MPC parameters in the cache: run `python main.py` first.")
    return {"cfg": cfg, "p": TRMSParams(), "nmpc": cache["nmpc"]["params"]}


def run_nmpc(p_model: TRMSParams, p_plant: TRMSParams, N: int, weights: dict, cfg: dict) -> dict:
    """One closed-loop evaluation run.

    ``p_model`` is what the N-MPC predicts with; ``p_plant`` is the simulated "true"
    TRMS. They are equal except in the robustness study. The run starts at the true
    plant's rest position.
    """
    sim, nm = cfg["simulation"], cfg["nmpc"]
    Ts, nsub = sim["Ts"], sim["plant_substeps"]
    sc = Scenario.from_config("evaluation", cfg["scenarios"]["evaluation"], rest_state(p_plant))
    ctrl = NMPCController(p_model, Ts, N, weights, nm["terminal_weight"], nm["prediction_substeps"],
                          nm["reference_preview"], nm.get("ipopt"), store_predictions=False)
    t0 = time.time()
    df = run_closed_loop(ctrl, sc, p_plant, Ts, nsub)
    wall = time.time() - t0
    n_expected = int(round(sc.duration / Ts))
    # same divergence limits as the tuning objective: |pitch| or |yaw error| too large,
    # or the simulation stopped early on a non-finite state
    obj = cfg["objective"]
    diverged = (len(df) < n_expected
                or bool(np.nanmax(np.abs(df["pitch"].values)) > obj["divergence_pitch_limit"])
                or bool(np.nanmax(np.abs(df["ref_yaw"].values - df["yaw"].values))
                        > obj["divergence_yaw_error_limit"]))
    _, g = controller_metrics(df, sc.steps(), Ts, u_limits=(p_plant.u_min, p_plant.u_max))
    g["diverged"] = diverged
    g["wall_time_s"] = wall
    ts = df[["time", "ref_pitch", "ref_yaw", "pitch", "yaw", "u_pitch", "u_yaw"]].copy()
    return {"metrics": g, "timeseries": ts}


def run_many(jobs: list[dict], workers: int | None = None) -> list[dict]:
    """Run ``run_nmpc(**job)`` for every job, in parallel worker processes."""
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_nmpc, **job) for job in jobs]
        return [f.result() for f in futs]


def output_dir(study: str, root: str | None = None) -> Path:
    cfg_root = root or ROOT / "outputs"
    d = Path(cfg_root) / "studies" / f"{study}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def markdown_table(df, fmt: dict | None = None) -> str:
    """Small dependency-free markdown table (pandas.to_markdown needs `tabulate`)."""
    fmt = fmt or {}

    def cell(col, v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "–"
            return format(v, fmt.get(col, ".3g"))
        return str(v)

    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [cell(c, row[c]) for c in df.columns]) + " |")
    return "\n".join(lines)
