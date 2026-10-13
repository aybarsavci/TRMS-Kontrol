"""Decentralised discrete PID (pitch <- main rotor, yaw <- tail rotor).

Per axis:  u = Kp*e + I + D,  with
  * derivative on measurement (no derivative kick on reference steps),
    filtered by a first-order low-pass with time constant ``tf``;
  * conditional-integration anti-windup (integrator frozen while the output is
    saturated and the error drives it further into saturation);
  * output clamped to the actuator limits.

The core ``pid_step`` is vectorised over a batch dimension so the auto-tuner can
simulate a whole optimiser population in one pass; ``PIDController`` wraps it
with batch size 1 for the logged simulation, so both use identical arithmetic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

GAIN_NAMES = ("Kp_pitch", "Ki_pitch", "Kd_pitch", "Kp_yaw", "Ki_yaw", "Kd_yaw")


@dataclass
class PIDState:
    integ: np.ndarray    # (2, B) integral term (already multiplied by Ki)
    d: np.ndarray        # (2, B) filtered derivative term
    y_prev: np.ndarray   # (2, B) previous measurement

    @classmethod
    def zeros(cls, y0: np.ndarray) -> "PIDState":
        y0 = np.atleast_2d(np.asarray(y0, float).reshape(2, -1))
        return cls(np.zeros_like(y0), np.zeros_like(y0), y0.copy())


def pid_step(gains, state: PIDState, y, r, Ts, tf, umin, umax):
    """One PID update. ``gains`` (6, B); ``y``, ``r`` (2, B). Returns (u, info)."""
    Kp, Ki, Kd = gains[[0, 3]], gains[[1, 4]], gains[[2, 5]]
    e = r - y
    P = Kp * e
    a = tf / (tf + Ts)
    D = a * state.d - (Kd / (tf + Ts)) * (y - state.y_prev)
    I_try = state.integ + Ki * Ts * e
    v_try = P + I_try + D
    freeze = ((v_try > umax) & (e > 0)) | ((v_try < umin) & (e < 0))
    I = np.where(freeze, state.integ, I_try)
    v = P + I + D
    u = np.clip(v, umin, umax)
    state.integ, state.d, state.y_prev = I, D, y.copy()
    return u, dict(e=e, P=P, I=I, D=D, v=v, freeze=freeze)


class PIDController:
    name = "PID"

    def __init__(self, gains: dict, Ts: float, tf: float, umin: float, umax: float):
        self.gains_dict = {k: float(gains[k]) for k in GAIN_NAMES}
        self.g = np.array([self.gains_dict[k] for k in GAIN_NAMES]).reshape(6, 1)
        self.Ts, self.tf, self.umin, self.umax = Ts, tf, umin, umax
        self.state = None

    def reset(self, x0):
        self.state = PIDState.zeros(np.array([x0[0], x0[3]]))

    def step(self, t, x, ref_now, ref_fn=None):
        t0 = time.perf_counter()
        y = np.array([[x[0]], [x[3]]])
        r = np.asarray(ref_now, float).reshape(2, 1)
        u, info = pid_step(self.g, self.state, y, r, self.Ts, self.tf, self.umin, self.umax)
        dt_ms = (time.perf_counter() - t0) * 1e3
        log = {"compute_time_ms": dt_ms}
        for i, ax in enumerate(("pitch", "yaw")):
            log[f"pid_P_{ax}"] = float(info["P"][i, 0])
            log[f"pid_I_{ax}"] = float(info["I"][i, 0])
            log[f"pid_D_{ax}"] = float(info["D"][i, 0])
            log[f"u_unsat_{ax}"] = float(info["v"][i, 0])
            log[f"pid_integrator_frozen_{ax}"] = bool(info["freeze"][i, 0])
        return u[:, 0], log

    def describe(self) -> dict:
        return {"type": "PID", "gains": self.gains_dict, "Ts": self.Ts,
                "derivative_filter_tf": self.tf, "anti_windup": "conditional integration"}
