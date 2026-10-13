"""Automatic parameter optimisation for both controllers.

Both controllers are tuned with the SAME objective (``metrics.tuning_objective``)
on the SAME tuning scenario, which is deliberately different from the evaluation
scenario so neither controller is scored on the exact trajectory it was fitted to.

* PID:   scipy differential evolution over log10 of the six gains. The whole
         population is simulated in one vectorised pass (``simulate_pid_batch``),
         which is ~50x faster than evaluating members one by one.
* N-MPC: scipy differential evolution over log10 of the Q/R weights and the
         integer prediction horizon N (``integrality``). Each candidate runs a
         full closed-loop IPOPT simulation; candidates run in parallel processes.
"""
from __future__ import annotations

import os
import time

import numpy as np
from scipy.optimize import differential_evolution

from .controllers.nmpc import WEIGHT_NAMES, NMPCController
from .controllers.pid import GAIN_NAMES
from .metrics import tuning_objective
from .params import TRMSParams
from .simulation import simulate_pid_batch


# ===================================================================== PID
def tune_pid(p, scenario, sim_cfg, obj_cfg, pid_cfg, verbose=True):
    Ts, nsub, tf = sim_cfg["Ts"], sim_cfg["plant_substeps"], pid_cfg["derivative_filter_tf"]
    bounds = [tuple(pid_cfg["bounds_log10"][k]) for k in GAIN_NAMES]
    steps = scenario.steps()
    de = pid_cfg["de"]
    history, nfev = [], [0]

    def obj(theta):                                    # theta (6, S)
        gains = 10.0 ** np.atleast_2d(theta.T).T
        t, ref, y, u = simulate_pid_batch(gains, scenario, p, Ts, nsub, tf)
        nfev[0] += gains.shape[1]
        return tuning_objective(t, ref, y, u, steps, obj_cfg, Ts)

    def cb(intermediate_result):
        history.append(float(intermediate_result.fun))
        if verbose:
            print(f"  [PID] generation {len(history):3d}  best J = {history[-1]:.5f}", flush=True)

    t0 = time.time()
    res = differential_evolution(
        obj, bounds, vectorized=True, updating="deferred", popsize=de["popsize"],
        maxiter=de["maxiter"], tol=de["tol"], seed=de["seed"], polish=False,
        mutation=(0.5, 1.0), recombination=0.7, callback=cb, init="sobol")
    gains = {k: float(10.0 ** v) for k, v in zip(GAIN_NAMES, res.x)}
    report = {"method": "scipy.optimize.differential_evolution (vectorised population)",
              "objective": float(res.fun), "n_function_evals": nfev[0],
              "generations": int(res.nit), "message": str(res.message),
              "wall_time_s": time.time() - t0, "convergence_history": history,
              "search_space": "log10 gains", "bounds_log10": dict(zip(GAIN_NAMES, bounds))}
    return gains, report


# ===================================================================== N-MPC
def nmpc_theta_to_params(theta, nmpc_cfg):
    w = {"q_pitch": float(nmpc_cfg["q_pitch"])}
    for name, v in zip(WEIGHT_NAMES[1:], theta[:5]):
        w[name] = float(10.0 ** v)
    return w, int(round(theta[5]))


def run_nmpc_scenario(p, scenario, Ts, nsub, weights, N, nmpc_cfg, obj_cfg=None):
    """Lightweight closed loop (no telemetry) used inside the tuner."""
    from .model import rk4_np
    ctrl = NMPCController(p, Ts, N, weights, nmpc_cfg["terminal_weight"],
                          nmpc_cfg["prediction_substeps"], nmpc_cfg["reference_preview"],
                          nmpc_cfg.get("ipopt"), store_predictions=False)
    n = int(round(scenario.duration / Ts))
    t = np.arange(n) * Ts
    ref = scenario.reference(t).T
    x = scenario.x0.copy()
    ctrl.reset(x)
    Y = np.full((n, 2), np.nan)
    U = np.zeros((n, 2))
    for k in range(n):
        u, _ = ctrl.step(t[k], x, ref[k], scenario.reference)
        Y[k], U[k] = x[[0, 3]], u
        x = rk4_np(x[:, None], u[:, None], Ts, p, nsub)[:, 0]
        if not np.all(np.isfinite(x)) or (obj_cfg and abs(x[0]) > obj_cfg["divergence_pitch_limit"]):
            Y[k + 1:] = Y[k]
            break
    return t, ref, Y, U


def _nmpc_objective(theta, ctx):
    p = TRMSParams(**{k: tuple(v) if isinstance(v, list) else v for k, v in ctx["params"].items()})
    from .scenario import Scenario
    sc = Scenario(**ctx["scenario"])
    w, N = nmpc_theta_to_params(theta, ctx["nmpc"])
    try:
        t, ref, y, u = run_nmpc_scenario(p, sc, ctx["Ts"], ctx["nsub"], w, N, ctx["nmpc"], ctx["obj"])
        J = float(tuning_objective(t, ref, y, u, sc.steps(), ctx["obj"], ctx["Ts"])[0])
    except Exception as exc:  # solver build/runtime failure -> heavily penalised
        print(f"  [N-MPC] candidate failed: {exc}", flush=True)
        J = ctx["obj"]["divergence_penalty"] * 10
    return J


def tune_nmpc(p, scenario, sim_cfg, obj_cfg, nmpc_cfg, verbose=True):
    b = nmpc_cfg["bounds"]
    bounds = [tuple(b[k]) for k in ("q_yaw_log10", "q_rate_pitch_log10", "q_rate_yaw_log10",
                                    "r_pitch_log10", "r_yaw_log10")] + [tuple(b["horizon_steps"])]
    ctx = {"params": p.to_dict(), "Ts": sim_cfg["Ts"], "nsub": sim_cfg["plant_substeps"],
           "obj": obj_cfg, "nmpc": nmpc_cfg,
           "scenario": {"name": scenario.name, "duration": scenario.duration,
                        "events": scenario.events, "x0": scenario.x0}}
    de = nmpc_cfg["de"]
    workers = de.get("workers", -1)
    if workers == -1:
        workers = os.cpu_count() or 1
    history = []
    t0 = time.time()

    def cb(intermediate_result):
        history.append(float(intermediate_result.fun))
        if verbose:
            w, N = nmpc_theta_to_params(intermediate_result.x, nmpc_cfg)
            print(f"  [N-MPC] generation {len(history):3d}  best J = {history[-1]:.5f}  "
                  f"(N={N})  elapsed {time.time() - t0:.0f}s", flush=True)

    if verbose:
        npop = de["popsize"] * len(bounds)
        print(f"  [N-MPC] population {npop}, up to {de['maxiter'] + 1} generations, "
              f"{workers} worker process(es)", flush=True)
    res = differential_evolution(
        _nmpc_objective, bounds, args=(ctx,), integrality=[0, 0, 0, 0, 0, 1],
        popsize=de["popsize"], maxiter=de["maxiter"], tol=de["tol"], seed=de["seed"],
        polish=False, updating="deferred", workers=workers, init="sobol",
        mutation=(0.5, 1.0), recombination=0.7, callback=cb)
    weights, N = nmpc_theta_to_params(res.x, nmpc_cfg)
    report = {"method": "scipy.optimize.differential_evolution (parallel, integer horizon)",
              "objective": float(res.fun), "n_function_evals": int(res.nfev),
              "generations": int(res.nit), "message": str(res.message),
              "wall_time_s": time.time() - t0, "convergence_history": history,
              "search_space": "log10 weights (q_pitch fixed for scale), integer N",
              "bounds": {k: list(v) for k, v in zip(
                  ["q_yaw_log10", "q_rate_pitch_log10", "q_rate_yaw_log10",
                   "r_pitch_log10", "r_yaw_log10", "horizon_steps"], bounds)}}
    return {"weights": weights, "horizon_steps": N}, report
