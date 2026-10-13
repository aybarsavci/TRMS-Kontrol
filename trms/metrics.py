"""Performance metrics and the shared tuning objective.

All functions accept arrays with time on axis 0 and an optional trailing batch
axis, so the same code scores one logged run or a whole optimiser population.
"""
from __future__ import annotations

import numpy as np


def _seg(t, s):
    return (t >= s["t_start"] - 1e-9) & (t < s["t_end"] - 1e-9)


_STEP_KEYS = ("overshoot_pct", "peak_value", "undershoot_pct", "rise_time_10_90_s",
              "delay_time_50_s", "settling_time_2pct_s", "steady_state_error", "IAE", "ISE", "ITAE")


def step_metrics(t, y, s, band=0.02):
    """Classic step-response metrics for one reference segment ``s`` (scalar y).

    A segment with fewer than two finite samples (a run that stopped early, or a
    step at the very end of a scenario) yields NaN metrics and ``settled=False``
    instead of raising on an empty array.
    """
    m = _seg(t, s)
    tt, yy = t[m], y[m]
    r0, r1, d = s["r_from"], s["r_to"], s["step_size"]
    out = dict(axis=s["axis"], t_start=s["t_start"], r_from=r0, r_to=r1, step_size=d)
    if len(yy) < 2 or not np.all(np.isfinite(yy)) or d == 0:
        out.update({k: np.nan for k in _STEP_KEYS})
        out["settled"] = False
        return out
    sgn = np.sign(d)
    prog = (yy - r0) / d                       # 0 -> start, 1 -> target (d != 0 checked above)

    over = max(0.0, float(np.max(sgn * (yy - r1))))
    out["overshoot_pct"] = 100.0 * over / abs(d)
    out["peak_value"] = float(yy[np.argmax(sgn * yy)])
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
    dt = np.diff(tt, append=tt[-1] + (tt[1] - tt[0]))
    out["IAE"] = float(np.sum(np.abs(e) * dt))
    out["ISE"] = float(np.sum(e ** 2 * dt))
    out["ITAE"] = float(np.sum((tt - s["t_start"]) * np.abs(e) * dt))
    out["settled"] = bool(np.isfinite(out["settling_time_2pct_s"]))
    return out


def global_metrics(t, ref, y, u, Ts, compute_ms=None, u_limits=(-1.0, 1.0)):
    """Whole-run metrics. ``ref``/``y``/``u``: (T, 2) arrays [pitch, yaw].

    Saturation is measured against the actual actuator limits ``u_limits``
    (previously a hard-coded |u| >= 0.999, wrong for non-unit or asymmetric limits).
    """
    umin, umax = u_limits
    span = umax - umin
    tol = 5e-4 * span                          # == 0.999 threshold for limits of +-1
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
        out[f"saturation_fraction_{ax}"] = float(np.mean((ui >= umax - tol) | (ui <= umin + tol)))
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


def tuning_objective(t, ref, y, u, steps, cfg, Ts, diverged=None):
    """Scalar cost used to tune BOTH controllers (lower is better).

    J = w_itae_pitch*ITAE_pitch + w_itae_yaw*ITAE_yaw
      + w_tv * (TV(u_pitch) + TV(u_yaw))
      + w_overshoot * sum_steps max(0, OS% - allowance)
      + divergence_penalty  (pitch beyond +-limit, yaw error beyond limit, blow-up,
                             a member flagged in ``diverged``, or a non-finite J)

    ITAE uses tau = time since the latest reference change, so error that
    persists (slow drift, limit cycles) costs far more than a fast transient.

    Shapes: t (T,), ref (T, 2), y/u (T, 2, B), diverged (B,) bool. Returns (B,).
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
        if not m.any() or s["step_size"] == 0:
            continue
        yi = y[m, s["axis_index"]]
        os = np.maximum(0.0, np.max(np.sign(s["step_size"]) * (yi - s["r_to"]), axis=0))
        os_pct = 100.0 * os / abs(s["step_size"])
        J = J + cfg["w_overshoot"] * np.maximum(0.0, os_pct - cfg["overshoot_allowance_pct"])
    with np.errstate(invalid="ignore"):
        bad = ~np.all(np.isfinite(y), axis=(0, 1)) \
            | ~np.all(np.isfinite(u), axis=(0, 1)) \
            | ~np.isfinite(J) \
            | np.any(np.abs(y[:, 0]) > cfg["divergence_pitch_limit"], axis=0) \
            | np.any(np.abs(e[:, 1]) > cfg["divergence_yaw_error_limit"], axis=0)
    if diverged is not None:
        bad = bad | np.asarray(diverged, bool)
    # BUG FIX: a non-finite J used to be replaced by 0 (a perfect score) WITHOUT the
    # penalty when y stayed finite, so a NaN candidate could win the optimisation.
    J = np.where(np.isfinite(J), J, 0.0)
    return np.where(bad, J + cfg["divergence_penalty"], J)
