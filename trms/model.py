"""Nonlinear TRMS dynamics (6 states, 2 inputs).

State  x = [alpha_v, S_v, u_vv, alpha_h, S_h, u_hh]
Input  u = [u_v, u_h]  (normalised motor voltages, [-1, 1])

    d alpha_v/dt = Omega_v = S_v + J_tr*w_t(u_hh)/J_v
    d S_v/dt     = [l_m S_f F_v(w_m) + g((A-B)cos a_v - C sin a_v)
                    - Omega_h^2 H sin a_v cos a_v - K_v Omega_v] / J_v
    d u_vv/dt    = (-u_vv + K_mr u_v) / T_mr
    d alpha_h/dt = Omega_h = S_h + J_mr*w_m(u_vv)*cos a_v / J_h(a_v)
    d S_h/dt     = [l_t S_f F_h(w_t) cos a_v - K_h Omega_h] / J_h(a_v)
    d u_hh/dt    = (-u_hh + K_tr u_h) / T_tr
    J_h(a_v)     = D cos^2 a_v + E sin^2 a_v + F

The same equations are evaluated with numpy (batched, shape (6, B)) for
simulation/tuning and with CasADi symbols for the N-MPC, so the plant and the
prediction model can never drift apart.
"""
from __future__ import annotations

import math
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq

from .params import TRMSParams

NX, NU = 6, 2


def polyval(coeffs, x):
    """Horner evaluation that works for floats, numpy arrays and CasADi symbols."""
    r = coeffs[0] * 1.0
    for c in coeffs[1:]:
        r = r * x + c
    return r


def terms(x, u, p: TRMSParams, sin, cos):
    """Return (xdot tuple, aux dict). ``x``/``u`` are indexable by state index."""
    a_v, S_v, u_vv, a_h, S_h, u_hh = (x[i] for i in range(NX))
    u_v, u_h = u[0], u[1]

    w_m = polyval(p.omega_m_poly, u_vv)
    w_t = polyval(p.omega_t_poly, u_hh)
    F_v = polyval(p.F_v_poly, w_m)
    F_h = polyval(p.F_h_poly, w_t)
    c, s = cos(a_v), sin(a_v)
    J_h = p.D * c * c + p.E * s * s + p.F

    Om_v = S_v + p.J_tr * w_t / p.J_v
    Om_h = S_h + p.J_mr * w_m * c / J_h

    M_aero_v = p.l_m * p.S_f * F_v
    M_grav = p.g * ((p.A - p.B) * c - p.C * s)
    M_cent = -(Om_h * Om_h) * p.H * s * c
    M_fric_v = -p.K_v * Om_v
    M_aero_h = p.l_t * p.S_f * F_h * c
    M_fric_h = -p.K_h * Om_h

    dS_v = (M_aero_v + M_grav + M_cent + M_fric_v) / p.J_v
    dS_h = (M_aero_h + M_fric_h) / J_h
    xdot = (
        Om_v,
        dS_v,
        (-u_vv + p.K_mr * u_v) / p.T_mr,
        Om_h,
        dS_h,
        (-u_hh + p.K_tr * u_h) / p.T_tr,
    )
    aux = dict(
        pitch_rate=Om_v, yaw_rate=Om_h,
        omega_main=w_m, omega_tail=w_t,
        thrust_main=p.S_f * F_v, thrust_tail=p.S_f * F_h,
        J_h=J_h,
        M_aero_v=M_aero_v, M_grav=M_grav, M_cent=M_cent, M_fric_v=M_fric_v,
        M_aero_h=M_aero_h, M_fric_h=M_fric_h,
        dS_v=dS_v, dS_h=dS_h,
    )
    return xdot, aux


# --------------------------------------------------------------------- numpy
def f_np(x: np.ndarray, u: np.ndarray, p: TRMSParams) -> np.ndarray:
    xdot, _ = terms(x, u, p, np.sin, np.cos)
    return np.stack(np.broadcast_arrays(*xdot))


def aux_np(x: np.ndarray, u: np.ndarray, p: TRMSParams) -> dict:
    _, aux = terms(x, u, p, np.sin, np.cos)
    return aux


def rk4_np(x, u, dt, p, substeps=1):
    """Integrate with zero-order-hold input over ``dt`` using ``substeps`` RK4 steps."""
    h = dt / substeps
    for _ in range(substeps):
        k1 = f_np(x, u, p)
        k2 = f_np(x + 0.5 * h * k1, u, p)
        k3 = f_np(x + 0.5 * h * k2, u, p)
        k4 = f_np(x + h * k3, u, p)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def rest_state(p: TRMSParams) -> np.ndarray:
    """Beam hanging at its gravitational equilibrium with both rotors off."""
    return np.array([p.rest_pitch, 0.0, 0.0, 0.0, 0.0, 0.0])


# --------------------------------------------------------------------- casadi
@lru_cache(maxsize=32)
def rk4_casadi(p: TRMSParams, dt: float, substeps: int = 1):
    """CasADi SX functions for one zero-order-hold sample: ``F(x, u) -> x_next`` and
    ``rates(x) -> [Omega_v, Omega_h]``. Built from the same ``terms`` as the numpy
    model. Cached: ``TRMSParams`` is frozen (hashable), so identical requests reuse
    the already-built expression graph instead of rebuilding it."""
    if not dt > 0 or int(substeps) < 1:
        raise ValueError(f"need dt > 0 and substeps >= 1 (got dt={dt}, substeps={substeps})")
    import casadi as ca

    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    ns = SimpleNamespace()

    def f(xx):
        xdot, _ = terms(xx, u, p, ca.sin, ca.cos)
        return ca.vertcat(*xdot)

    h = dt / substeps
    xk = x
    for _ in range(substeps):
        k1 = f(xk)
        k2 = f(xk + 0.5 * h * k1)
        k3 = f(xk + 0.5 * h * k2)
        k4 = f(xk + h * k3)
        xk = xk + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    ns.F = ca.Function("F_rk4", [x, u], [xk], ["x", "u"], ["x_next"])
    _, aux = terms(x, u, p, ca.sin, ca.cos)
    ns.rates = ca.Function("rates", [x], [ca.vertcat(aux["pitch_rate"], aux["yaw_rate"])])
    return ns


class Plant:
    """The simulated "true" plant: RK4 with ``substeps`` steps per sample, evaluated by
    CasADi's SX virtual machine instead of ~40 numpy calls on tiny arrays per sample.

    Same equations, same operation order as ``rk4_np``: results agree to the last bit
    (checked in tests), while one sample costs ~40 us instead of ~3 ms.

    ``step(x, u)`` accepts one state (6,) or a batch (6, B) / (2, B); a batch is
    evaluated with ``Function.map`` (one compiled loop over the columns).
    """

    def __init__(self, p: TRMSParams, dt: float, substeps: int):
        self.F = rk4_casadi(p, float(dt), int(substeps)).F
        self._maps = {}

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return np.asarray(self.F(x, u)).ravel()
        B = x.shape[1]
        fm = self._maps.get(B)
        if fm is None:
            fm = self._maps[B] = self.F.map(B, "serial")
        return np.asarray(fm(x, u))


# --------------------------------------------------------------------- trim
def trim_main_input(pitch: float, p: TRMSParams) -> float:
    """Main-rotor input that holds ``pitch`` at rest (yaw rotor off). NaN if infeasible."""
    def residual(uv):
        x = np.array([pitch, 0.0, uv, 0.0, 0.0, 0.0])
        return float(f_np(x, np.array([uv, 0.0]), p)[1])

    lo, hi = p.u_min, p.u_max
    if residual(lo) * residual(hi) > 0:
        return math.nan
    return brentq(residual, lo, hi, xtol=1e-10)


def pitch_envelope(p: TRMSParams, n: int = 200):
    """Table of (pitch, trim input) over the physically reachable range."""
    pitches = np.linspace(-1.2, 1.0, n)
    trims = np.array([trim_main_input(a, p) for a in pitches])
    return pitches, trims
