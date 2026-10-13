"""Closed-loop simulation with exhaustive telemetry.

The plant is integrated with RK4 at Ts/plant_substeps under a zero-order-hold
input. Each control sample logs the reference, every plant state, every derived
physical quantity (rotor speeds, thrusts, torque contributions, accelerations,
inertia), commanded and applied inputs, and controller internals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .controllers.pid import PIDState, pid_step
from .model import aux_np, f_np, rk4_np
from .params import STATE_NAMES


def run_closed_loop(controller, scenario, p, Ts, plant_substeps, progress=None):
    n = int(round(scenario.duration / Ts))
    x = scenario.x0.astype(float).copy()
    controller.reset(x)
    rows = []
    u_prev = np.zeros(2)
    iae = np.zeros(2)
    eff = np.zeros(2)
    for k in range(n):
        t = k * Ts
        ref = scenario.reference(t)
        u, clog = controller.step(t, x, ref, scenario.reference)
        u = np.asarray(u, float)

        xdot = f_np(x[:, None], u[:, None], p)[:, 0]
        aux = {k2: float(np.asarray(v).ravel()[0]) for k2, v in aux_np(x[:, None], u[:, None], p).items()}
        e = ref - np.array([x[0], x[3]])
        iae += np.abs(e) * Ts
        eff += u ** 2 * Ts

        row = {"time": t, "ref_pitch": ref[0], "ref_yaw": ref[1]}
        for i, nm in enumerate(STATE_NAMES):
            row[f"x_{nm}"] = x[i]
            row[f"xdot_{nm}"] = xdot[i]
        row.update({
            "pitch": x[0], "yaw": x[3],
            "pitch_deg": np.degrees(x[0]), "yaw_deg": np.degrees(x[3]),
            "ref_pitch_deg": np.degrees(ref[0]), "ref_yaw_deg": np.degrees(ref[1]),
            "err_pitch": e[0], "err_yaw": e[1],
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
            "u_pitch": u[0], "u_yaw": u[1],
            "du_pitch": u[0] - u_prev[0], "du_yaw": u[1] - u_prev[1],
            "sat_pitch": bool(abs(u[0]) >= p.u_max - 1e-9),
            "sat_yaw": bool(abs(u[1]) >= p.u_max - 1e-9),
            "cum_IAE_pitch": iae[0], "cum_IAE_yaw": iae[1],
            "cum_int_u2_pitch": eff[0], "cum_int_u2_yaw": eff[1],
        })
        row.update(clog)
        rows.append(row)

        x = rk4_np(x[:, None], u[:, None], Ts, p, plant_substeps)[:, 0]
        u_prev = u
        if not np.all(np.isfinite(x)):
            break
        if progress and k % max(1, n // 20) == 0:
            progress(k / n)
    return pd.DataFrame(rows)


def simulate_pid_batch(gains, scenario, p, Ts, plant_substeps, tf):
    """Simulate a population of PID gain sets at once. ``gains`` (6, B).

    Returns t (T,), ref (T, 2), y (T, 2, B), u (T, 2, B).
    """
    B = gains.shape[1]
    n = int(round(scenario.duration / Ts))
    t = np.arange(n) * Ts
    ref = scenario.reference(t).T                       # (T, 2)
    x = np.tile(scenario.x0.reshape(6, 1), (1, B))
    st = PIDState.zeros(np.tile(np.array([[x[0, 0]], [x[3, 0]]]), (1, B)))
    Y = np.empty((n, 2, B))
    U = np.empty((n, 2, B))
    with np.errstate(all="ignore"):
        for k in range(n):
            y = x[[0, 3]]
            u, _ = pid_step(gains, st, y, np.repeat(ref[k][:, None], B, 1), Ts, tf, p.u_min, p.u_max)
            Y[k], U[k] = y, u
            x = rk4_np(x, u, Ts, p, plant_substeps)
            x = np.where(np.isfinite(x), x, 1e3)       # keep diverged members bounded
    return t, ref, Y, U
