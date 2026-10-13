"""TRMS reference tracking: N-MPC vs PID.

Usage
-----
    python main.py                    # use cached tuned parameters (tune only if missing/stale)
    python main.py --retune all       # force re-optimisation of both controllers
    python main.py --retune pid       # force PID only (or: nmpc)
    python main.py --quick            # small search budget smoke test (cached separately)
"""
from __future__ import annotations

import argparse
import copy
import time
from datetime import datetime
from pathlib import Path

import yaml

from trms.cache import ParameterCache, signature
from trms.controllers.nmpc import NMPCController
from trms.controllers.pid import PIDController
from trms.model import pitch_envelope, rest_state
from trms.params import TRMSParams
from trms.plotting import plot_comparison, plot_convergence, plot_single_run
from trms.reporting import (controller_metrics, write_comparison, write_controller_folder,
                            write_summary)
from trms.scenario import Scenario
from trms.simulation import run_closed_loop
from trms.tuning import tune_nmpc, tune_pid

ROOT = Path(__file__).resolve().parent


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retune", choices=["pid", "nmpc", "all"], default=None,
                    help="force re-optimisation even if valid cached parameters exist")
    ap.add_argument("--quick", action="store_true", help="reduced search budget (smoke test)")
    ap.add_argument("--settings", default=str(ROOT / "config" / "settings.yaml"))
    ap.add_argument("--cache", default=None,
                    help="cache file (default config/tuned_parameters.json, "
                         "or config/tuned_parameters_quick.json with --quick)")
    ap.add_argument("--output-root", default=None, help="override output.root from settings")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.settings).read_text())
    if args.quick:
        cfg = copy.deepcopy(cfg)
        cfg["pid"]["de"].update(cfg["quick"]["pid_de"])
        cfg["nmpc"]["de"].update(cfg["quick"]["nmpc_de"])
    p = TRMSParams()
    sim, obj = cfg["simulation"], cfg["objective"]
    Ts, nsub = sim["Ts"], sim["plant_substeps"]
    x0 = rest_state(p)
    sc_tune = Scenario.from_config("tuning", cfg["scenarios"]["tuning"], x0)
    sc_eval = Scenario.from_config("evaluation", cfg["scenarios"]["evaluation"], x0)
    # --quick results live in their own file so they never overwrite the full-budget cache
    default_cache = "tuned_parameters_quick.json" if args.quick else "tuned_parameters.json"
    cache = ParameterCache(args.cache or (ROOT / "config" / default_cache))
    base_sig = {"model": p.to_dict(), "simulation": sim, "objective": obj,
                "tuning_scenario": cfg["scenarios"]["tuning"]}

    # ------------------------------------------------------------- tuning
    entries = {}
    for key, section, tuner in (("pid", "pid", tune_pid), ("nmpc", "nmpc", tune_nmpc)):
        sig = signature({**base_sig, "controller": cfg[section]})
        cached = cache.get(key, sig)
        force = args.retune in (key, "all")
        if cached and not force:
            print(f"[{key.upper()}] using cached parameters from {cache.path.name} "
                  f"(tuned {cached['created_utc']})")
            entries[key] = cached
            continue
        why = "forced by --retune" if force else ("settings changed" if key in cache.data else "no cache")
        print(f"[{key.upper()}] auto-tuning ({why}) ...")
        params, report = tuner(p, sc_tune, sim, obj, cfg[section])
        cache.put(key, sig, params, report)
        entries[key] = cache.data[key]
        print(f"[{key.upper()}] done in {report['wall_time_s']:.0f}s, J = {report['objective']:.5f}")

    pid_params = entries["pid"]["params"]
    nmpc_params = entries["nmpc"]["params"]
    controllers = {
        "PID": PIDController(pid_params, Ts, cfg["pid"]["derivative_filter_tf"], p.u_min, p.u_max),
        "N-MPC": NMPCController(p, Ts, nmpc_params["horizon_steps"], nmpc_params["weights"],
                                cfg["nmpc"]["terminal_weight"], cfg["nmpc"]["prediction_substeps"],
                                cfg["nmpc"]["reference_preview"], cfg["nmpc"].get("ipopt")),
    }
    entry_by_name = {"PID": entries["pid"], "N-MPC": entries["nmpc"]}

    # ------------------------------------------------------------- outputs
    out_root = Path(args.output_root or (ROOT / cfg["output"]["root"]))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_root / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = sc_eval.steps()
    env = pitch_envelope(p)

    dfs, step_tables, globals_ = {}, {}, {}
    for name, ctrl in controllers.items():
        print(f"[{name}] simulating evaluation scenario ({sc_eval.duration:.0f} s) ...")
        t0 = time.time()
        df = run_closed_loop(ctrl, sc_eval, p, Ts, nsub)
        print(f"[{name}] {len(df)} samples in {time.time() - t0:.1f}s")
        sdf, g = controller_metrics(df, steps, Ts)
        folder = run_dir / name
        preds = getattr(ctrl, "predictions", None)
        write_controller_folder(folder, name, df, sdf, g, ctrl.describe(), entry_by_name[name], preds)
        plot_single_run(df, name, folder, steps, envelope=env, predictions=preds)
        plot_convergence(entry_by_name[name]["tuning_report"].get("convergence_history"), name, folder)
        dfs[name], step_tables[name], globals_[name] = df, sdf, g

    params_doc = {"model": p.to_dict(), "settings": cfg,
                  "controllers": {n: c.describe() for n, c in controllers.items()}}
    step_df, glob_df, wide = write_comparison(run_dir, dfs, step_tables, globals_, params_doc)
    plot_comparison(dfs, step_df, glob_df, steps, run_dir)
    write_summary(run_dir, glob_df, wide, params_doc,
                  {"run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
                   "created": datetime.now().isoformat(timespec="seconds"),
                   "scenario": ", ".join(f"{a} {v:+.2f} rad @ {t:.0f}s" for a, t, v in sc_eval.events)})
    (run_dir / "settings_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    print("\n" + (run_dir / "summary.md").read_text().split("## Per-step")[0])
    print(f"All outputs written to: {run_dir}")


if __name__ == "__main__":
    main()
