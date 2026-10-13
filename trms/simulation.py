"""Closed-loop simulation with exhaustive telemetry.

The plant is integrated with RK4 at Ts/plant_substeps under a zero-order-hold
input. Each control sample logs the reference, every plant state, every derived
physical quantity (rotor speeds, thrusts, torque contributions, accelerations,
inertia), commanded and applied inputs, and controller internals.

Performance design
------------------
* All per-sample storage is pre-allocated (states, inputs, controller logs); the
  control loop only writes into arrays. Nothing is appended inside the loop.
* The loop does only what must be sequential: controller step + plant step.
  Every derived quantity (derivatives, rotor speeds, torques, cumulative
  integrals, saturation flags) is computed afterwards in ONE vectorised pass over
  the whole (6, T) trajectory with the same ``model.terms`` equations.
* The plant step is a CasADi SX function (``model.Plant``): bit-identical to the
  numpy RK4, ~70x faster for a single trajectory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .controllers.pid import PIDState, check_timing, pid_step
from .model import Plant, aux_np, f_np
from .params import STATE_NAMES


class _LogBuffer:
    """Column store for controller logs, allocated on the first sample."""

    def __init__(self, n: int):
        self.n = n
        self.cols = None

    def write(self, k: int, log: dict):
        if self.cols is None:
            self.cols = {}
            for key, v in log.items():
                if isinstance(v, (bool, np.bool_)):
                    self.cols[key] = np.zeros(self.n, dtype=bool)
                elif isinstance(v, (int, np.integer)):
                    self.cols[key] = np.zeros(self.n, dtype=np.int64)
                elif isinstance(v, str):
                    self.cols[key] = np.empty(self.n, dtype=object)
                else:
                    self.cols[key] = np.full(self.n, np.nan)
        for key, v in log.items():
            self.cols[key][k] = v

    def trimmed(self, m: int) -> dict:
        return {key: col[:m] for key, col in (self.cols or {}).items()}


def run_closed_loop(controller, scenario, p, Ts, plant_substeps, progress=None, plant=None):
    check_timing(Ts, 0.0)
    n = int(round(scenario.duration / Ts))
    t = np.arange(n) * Ts
    R = scenario.reference(t).T                      # (n, 2), whole reference at once
    plant = plant or Plant(p, Ts, plant_substeps)

    X = np.empty((n, 6))
    U = np.empty((n, 2))
    logs = _LogBuffer(n)
    x = scenario.x0.astype(float).copy()
    controller.reset(x, n_steps=n)
    m = n
    for k in range(n):
        u, clog = controller.step(t[k], x, R[k], scenario.reference)
        u = np.asarray(u, float)
        X[k], U[k] = x, u
        logs.write(k, clog)
        x = plant.step(x, u)
        if not np.all(np.isfinite(x)):
            m = k + 1
            print(f"[{getattr(controller, 'name', 'controller')}] state became non-finite "
                  f"after t = {t[k]:.2f} s; simulation stopped early.")
            break
        if progress and k % max(1, n // 20) == 0:
            progress(k / n)

    return _telemetry_frame(t[:m], R[:m], X[:m], U[:m], logs.trimmed(m), p, Ts)


def _telemetry_frame(t, R, X, U, clogs, p, Ts) -> pd.DataFrame:
    """Build the telemetry table from whole-trajectory arrays (vectorised)."""
    xs, us = X.T, U.T                                 # (6, m), (2, m)
    xdot = f_np(xs, us, p)                            # (6, m)
    aux = aux_np(xs, us, p)
    E = R - X[:, [0, 3]]
    u_prev = np.vstack([np.zeros((1, 2)), U[:-1]])

    cols = {"time": t, "ref_pitch": R[:, 0], "ref_yaw": R[:, 1]}
    for i, nm in enumerate(STATE_NAMES):
        cols[f"x_{nm}"] = X[:, i]
        cols[f"xdot_{nm}"] = xdot[i]
    cols.update({
        "pitch": X[:, 0], "yaw": X[:, 3],
        "pitch_deg": np.degrees(X[:, 0]), "yaw_deg": np.degrees(X[:, 3]),
        "ref_pitch_deg": np.degrees(R[:, 0]), "ref_yaw_deg": np.degrees(R[:, 1]),
        "err_pitch": E[:, 0], "err_yaw": E[:, 1],
        "pitch_rate": aux["pitch_rate"], "yaw_rate": aux["yaw_rate"],
        "pitch_acc": aux["dS_v"], "yaw_acc": aux["dS_h"],
        "omega_main_rad_s": aux["omega_main"], "omega_tail_rad_s": aux["omega_tail"],
        "omega_main_rpm": aux["omega_main"] * 30 / np.pi,
        "omega_tail_rpm": aux["omega_tail"] * 30 / np.pi,
        "thrust_main_N": aux["thrust_main"], "thrust_tail_N": aux["thrust_tail"],
        "J_h": aux["J_h"],
        "torque_aero_pitch": aux["M_aero_v"], "torque_gravity_pitch": aux["M_grav"],
        "torque_centrifugal_pitch": aux["M_cent"], "torque_friction_pitch": aux["M_fric_v"],
        "torque_aero_yaw": aux["M_aero_h"], "torque_friction_yaw": aux["M_fric_h"],
        "u_pitch": U[:, 0], "u_yaw": U[:, 1],
        "du_pitch": U[:, 0] - u_prev[:, 0], "du_yaw": U[:, 1] - u_prev[:, 1],
        # saturation against the actual limits (symmetric or not)
        "sat_pitch": (U[:, 0] >= p.u_max - 1e-9) | (U[:, 0] <= p.u_min + 1e-9),
        "sat_yaw": (U[:, 1] >= p.u_max - 1e-9) | (U[:, 1] <= p.u_min + 1e-9),
        # sequential running sums: same additions in the same order as a loop
        "cum_IAE_pitch": np.cumsum(np.abs(E[:, 0]) * Ts),
        "cum_IAE_yaw": np.cumsum(np.abs(E[:, 1]) * Ts),
        "cum_int_u2_pitch": np.cumsum(U[:, 0] ** 2 * Ts),
        "cum_int_u2_yaw": np.cumsum(U[:, 1] ** 2 * Ts),
    })
    cols.update(clogs)
    return pd.DataFrame({k: np.broadcast_to(v, t.shape) if np.ndim(v) == 0 else v
                         for k, v in cols.items()})


def simulate_pid_batch(gains, scenario, p, Ts, plant_substeps, tf, plant=None):
    """Simulate a population of PID gain sets at once. ``gains`` (6, B).

    Returns t (T,), ref (T, 2), y (T, 2, B), u (T, 2, B), diverged (B,) bool.
    A member whose state becomes non-finite is marked diverged and frozen at its
    last finite state, so NaN/inf never propagates into the objective.
    """
    check_timing(Ts, tf)
    B = gains.shape[1]
    n = int(round(scenario.duration / Ts))
    t = np.arange(n) * Ts
    ref = scenario.reference(t).T                       # (T, 2)
    plant = plant or Plant(p, Ts, plant_substeps)
    x = np.tile(scenario.x0.reshape(6, 1), (1, B))
    st = PIDState.zeros(np.tile(np.array([[x[0, 0]], [x[3, 0]]]), (1, B)))
    Y = np.empty((n, 2, B))
    U = np.empty((n, 2, B))
    diverged = np.zeros(B, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(n):
            y = x[[0, 3]]
            u, _ = pid_step(gains, st, y, ref[k][:, None], Ts, tf, p.u_min, p.u_max)
            bad_u = ~np.all(np.isfinite(u), axis=0)
            if bad_u.any():                                 # flag it, keep the batch finite
                diverged |= bad_u
                u = np.where(np.isfinite(u), u, 0.0)
            Y[k], U[k] = y, u
            x_new = plant.step(x, u)
            bad = ~np.all(np.isfinite(x_new), axis=0)
            if bad.any():
                diverged |= bad
                x_new[:, bad] = x[:, bad]
            x = x_new
    return t, ref, Y, U, diverged
