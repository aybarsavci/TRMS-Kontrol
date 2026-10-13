"""Performance metrics and the shared tuning objective.

All functions accept arrays with time on axis 0 and an optional trailing batch
axis, so the same code scores one logged run or a whole optimiser population.
"""
from __future__ import annotations

import numpy as np


def _seg(t, s):
    return (t >= s["t_start"] - 1e-9) & (t < s["t_end"] - 1e-9)


def step_metrics(t, y, s, band=0.02):
    """Classic step-response metrics for one reference segment ``s`` (scalar y)."""
    m = _seg(t, s)
    tt, yy = t[m], y[m]
    r0, r1, d = s["r_from"], s["r_to"], s["step_size"]
    sgn = np.sign(d)
    prog = (yy - r0) / d                       # 0 -> start, 1 -> target
    out = dict(axis=s["axis"], t_start=s["t_start"], r_from=r0, r_to=r1, step_size=d)

    over = max(0.0, float(np.max(sgn * (yy - r1)))) if len(yy) else np.nan
    out["overshoot_pct"] = 100.0 * over / abs(d)
    out["peak_value"] = float(yy[np.argmax(sgn * yy)]) if len(yy) else np.nan
    out["undershoot_pct"] = 100.0 * max(0.0, float(np.max(-sgn * (yy - r0)))) / abs(d)

    def first_cross(level):
        idx = np.flatnonzero(prog >= level)
        return float(tt[idx[0]] - s["t_start"]) if idx.size else np.nan

    t10, t90 = first_cross(0.1), first_cross(0.9)
    out["rise_time_10_90_s"] = t90 - t10 if np.isfinite(t10) and np.isfinite(t90) else np.nan
    out["delay_time_50_s"] = first_cross(0.5)

    tol = band * abs(d)
    outside = np.flatnonzero(np.abs(yy - r1) > tol)
    if outside.size == 0:
        out["settling_time_2pct_s"] = 0.0
    elif outside[-1] == len(yy) - 1:
        out["settling_time_2pct_s"] = np.nan           # never settled in segment
    else:
        out["settling_time_2pct_s"] = float(tt[outside[-1] + 1] - s["t_start"])
    tail = tt >= tt[-1] - 1.0
    out["steady_state_error"] = float(np.mean(r1 - yy[tail]))
    e = r1 - yy
    dt = np.diff(tt, append=tt[-1] + (tt[1] - tt[0] if len(tt) > 1 else 0))
    out["IAE"] = float(np.sum(np.abs(e) * dt))
    out["ISE"] = float(np.sum(e ** 2 * dt))
    out["ITAE"] = float(np.sum((tt - s["t_start"]) * np.abs(e) * dt))
    out["settled"] = bool(np.isfinite(out["settling_time_2pct_s"]))
    return out


def global_metrics(t, ref, y, u, Ts, compute_ms=None):
    """Whole-run metrics. ``ref``/``y``/``u``: (T, 2) arrays [pitch, yaw]."""
    out = {}
    e = ref - y
    tw = t - t[0]
    for i, ax in enumerate(("pitch", "yaw")):
        out[f"IAE_{ax}"] = float(np.sum(np.abs(e[:, i])) * Ts)
        out[f"ISE_{ax}"] = float(np.sum(e[:, i] ** 2) * Ts)
        out[f"ITAE_{ax}"] = float(np.sum(tw * np.abs(e[:, i])) * Ts)
        out[f"RMSE_{ax}"] = float(np.sqrt(np.mean(e[:, i] ** 2)))
        out[f"max_abs_error_{ax}"] = float(np.max(np.abs(e[:, i])))
        ui = u[:, i]
        out[f"effort_int_u2_{ax}"] = float(np.sum(ui ** 2) * Ts)
        out[f"effort_int_abs_u_{ax}"] = float(np.sum(np.abs(ui)) * Ts)
        out[f"total_variation_u_{ax}"] = float(np.sum(np.abs(np.diff(ui))))
        out[f"max_abs_du_{ax}"] = float(np.max(np.abs(np.diff(ui)))) if len(ui) > 1 else 0.0
        out[f"mean_u_{ax}"] = float(np.mean(ui))
        out[f"rms_u_{ax}"] = float(np.sqrt(np.mean(ui ** 2)))
        out[f"saturation_fraction_{ax}"] = float(np.mean(np.abs(ui) >= 0.999))
    out["IAE_total"] = out["IAE_pitch"] + out["IAE_yaw"]
    out["effort_int_u2_total"] = out["effort_int_u2_pitch"] + out["effort_int_u2_yaw"]
    out["total_variation_u_total"] = out["total_variation_u_pitch"] + out["total_variation_u_yaw"]
    if compute_ms is not None:
        c = np.asarray(compute_ms, float)
        out["compute_time_mean_ms"] = float(np.mean(c))
        out["compute_time_median_ms"] = float(np.median(c))
        out["compute_time_p95_ms"] = float(np.percentile(c, 95))
        out["compute_time_max_ms"] = float(np.max(c))
        out["realtime_budget_exceeded_fraction"] = float(np.mean(c > Ts * 1e3))
    return out


def time_since_last_change(t, steps):
    """tau(t) = time elapsed since the most recent reference change on any axis."""
    starts = np.array(sorted({0.0} | {s["t_start"] for s in steps}))
    idx = np.searchsorted(starts, t + 1e-9, side="right") - 1
    return t - starts[np.clip(idx, 0, None)]


def tuning_objective(t, ref, y, u, steps, cfg, Ts):
    """Scalar cost used to tune BOTH controllers (lower is better).

    J = w_itae_pitch*ITAE_pitch + w_itae_yaw*ITAE_yaw
      + w_tv * (TV(u_pitch) + TV(u_yaw))
      + w_overshoot * sum_steps max(0, OS% - allowance)
      + divergence_penalty  (pitch beyond +-limit, yaw error beyond limit, or blow-up)

    ITAE uses tau = time since the latest reference change, so error that
    persists (slow drift, limit cycles) costs far more than a fast transient.

    Shapes: t (T,), ref (T, 2), y/u (T, 2, B). Returns (B,).
    """
    if y.ndim == 2:
        y, u = y[..., None], u[..., None]
    e = ref[..., None] - y
    tau = time_since_last_change(t, steps)[:, None]
    J = cfg["w_itae_pitch"] * np.sum(tau * np.abs(e[:, 0]), axis=0) * Ts \
        + cfg["w_itae_yaw"] * np.sum(tau * np.abs(e[:, 1]), axis=0) * Ts
    J = J + cfg["w_tv"] * np.sum(np.abs(np.diff(u, axis=0)), axis=(0, 1))
    for s in steps:
        m = _seg(t, s)
        yi = y[m, s["axis_index"]]
        os = np.maximum(0.0, np.max(np.sign(s["step_size"]) * (yi - s["r_to"]), axis=0))
        os_pct = 100.0 * os / abs(s["step_size"])
        J = J + cfg["w_overshoot"] * np.maximum(0.0, os_pct - cfg["overshoot_allowance_pct"])
    bad = ~np.all(np.isfinite(y), axis=(0, 1)) \
        | np.any(np.abs(y[:, 0]) > cfg["divergence_pitch_limit"], axis=0) \
        | np.any(np.abs(e[:, 1]) > cfg["divergence_yaw_error_limit"], axis=0)
    J = np.where(np.isfinite(J), J, 0.0)
    return np.where(bad, J + cfg["divergence_penalty"], J)
