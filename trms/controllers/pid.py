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
    """One PID update. ``gains`` (6, B); ``y`` (2, B); ``r`` (2, B) or (2, 1). Returns (u, info).

    Anti-windup is two-layered:
      1. conditional integration: the integrator is frozen while the output is
         saturated and the error would drive it further into saturation;
      2. integrator clamp: the integral term alone may never exceed the actuator
         range. Layer 1 cannot stop I from growing while a large opposite P or D
         keeps the *sum* inside the limits; the clamp bounds that case too.
    ``tf + Ts`` is the only divisor; ``check_timing`` guarantees it is > 0.
    """
    Kp, Ki, Kd = gains[[0, 3]], gains[[1, 4]], gains[[2, 5]]
    e = r - y
    P = Kp * e
    a = tf / (tf + Ts)
    D = a * state.d - (Kd / (tf + Ts)) * (y - state.y_prev)
    I_old = state.integ
    I_try = I_old + Ki * Ts * e
    v_try = P + I_try + D
    hi = (v_try > umax) & (e > 0)
    lo = (v_try < umin) & (e < 0)
    freeze = hi | lo
    # Conditional integration. BUG FIX: the old rule kept I_old whenever the tentative
    # output crossed a limit, so if ONE integration step jumped past the limit the
    # integrator never moved and the output stayed short of saturation (with a large
    # Ki it could stay stuck at 0 under a large error). Now the integrator advances
    # up to the value that puts the output exactly on the limit, never beyond it,
    # and never backwards (it is still frozen once the output already saturates).
    I_hi = np.maximum(I_old, np.minimum(I_try, umax - P - D))
    I_lo = np.minimum(I_old, np.maximum(I_try, umin - P - D))
    I = np.where(hi, I_hi, np.where(lo, I_lo, I_try))
    I = np.clip(I, umin, umax)
    v = P + I + D
    u = np.clip(v, umin, umax)
    state.integ, state.d, state.y_prev = I, D, y
    return u, dict(e=e, P=P, I=I, D=D, v=v, freeze=freeze)


def check_timing(Ts: float, tf: float):
    if not Ts > 0:
        raise ValueError(f"sample time Ts must be > 0 (got {Ts})")
    if not tf >= 0:
        raise ValueError(f"derivative filter time constant must be >= 0 (got {tf})")


class PIDController:
    name = "PID"

    def __init__(self, gains: dict, Ts: float, tf: float, umin: float, umax: float):
        check_timing(Ts, tf)
        self.gains_dict = {k: float(gains[k]) for k in GAIN_NAMES}
        bad = [k for k, v in self.gains_dict.items() if not np.isfinite(v)]
        if bad:
            raise ValueError(f"non-finite PID gains: {bad}")
        self.g = np.array([self.gains_dict[k] for k in GAIN_NAMES]).reshape(6, 1)
        self.Ts, self.tf, self.umin, self.umax = Ts, tf, umin, umax
        self.state = None

    def reset(self, x0, n_steps: int | None = None):
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
                "derivative_filter_tf": self.tf,
                "anti_windup": "conditional integration + integrator clamp to actuator range"}
