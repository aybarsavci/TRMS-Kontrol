"""Nonlinear MPC built on CasADi + IPOPT (multiple shooting).

At every sample the controller solves

  min  sum_{k=1..N} w_k [ q_p e_p,k^2 + q_y e_y,k^2 + q_dp Omega_v,k^2 + q_dy Omega_h,k^2 ]
     + sum_{k=0..N-1}   [ r_p (du_p,k)^2 + r_y (du_y,k)^2 ]
  s.t. x_{k+1} = F_RK4(x_k, u_k),  x_0 = measured state,  u_min <= u_k <= u_max

with w_k = 1 except w_N = ``terminal_weight`` (terminal penalty) and
du_k = u_k - u_{k-1} (u_{-1} = previously applied input). The prediction model is
the exact plant equations from ``trms.model`` (same code, CasADi symbols).

Q = diag(q_p, q_y, q_dp, q_dy) on [pitch error, yaw error, pitch rate, yaw rate],
R = diag(r_p, r_y) on input increments, horizon N - these are the tuned values.
Full-state feedback is assumed (the plant state is available to the controller).

Performance notes
-----------------
* The weights enter the NLP as *parameters*, not constants. The expression graph,
  its exact derivatives (automatic differentiation) and the IPOPT instance therefore
  depend only on (model, Ts, N, terminal weight, substeps, solver options) and are
  built once per process and reused (``_build_solver`` is cached). The tuner, which
  tries hundreds of weight sets, rebuilds a solver only when the horizon N changes.
* Warm start: the previous primal solution AND its multipliers are shifted by one
  sample, so both guesses describe the same time steps (previously the multipliers
  were passed unshifted, inconsistent with the shifted primal guess).
* Diagonal Q and R are applied element-wise; no dense matrix products or inversions.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache

import numpy as np

from ..model import NU, NX, rk4_casadi

WEIGHT_NAMES = ("q_pitch", "q_yaw", "q_rate_pitch", "q_rate_yaw", "r_pitch", "r_yaw")
NW = len(WEIGHT_NAMES)
# nlpsol options that belong to CasADi itself; any other key without a dot is an IPOPT option
_CASADI_KEYS = {"print_time", "expand", "jit", "compiler", "jit_options", "verbose", "record_time"}


def _solver_options(ipopt_options: dict | None) -> dict:
    opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes",
            "ipopt.max_iter": 100, "ipopt.tol": 1e-6, "ipopt.acceptable_tol": 1e-4,
            "expand": True}
    for k, v in (ipopt_options or {}).items():
        opts[k if ("." in k or k in _CASADI_KEYS) else f"ipopt.{k}"] = v
    return opts


@lru_cache(maxsize=32)
def _build_solver(p, Ts: float, N: int, terminal_weight: float, substeps: int, opts_json: str):
    """Build the multiple-shooting NLP and its IPOPT solver once per configuration.

    Parameter vector P = [x0 (6), R (2N, stage-major), u_prev (2), weights (6)].
    """
    import casadi as ca

    mdl = rk4_casadi(p, Ts, substeps)
    X = ca.SX.sym("X", NX, N + 1)
    U = ca.SX.sym("U", NU, N)
    P = ca.SX.sym("P", NX + NU * N + NU + NW)
    x0 = P[:NX]
    R = ca.reshape(P[NX:NX + NU * N], NU, N)
    u_prev = P[NX + NU * N:NX + NU * N + NU]
    q_p, q_y, q_dp, q_dy, r_p, r_y = (P[NX + NU * N + NU + i] for i in range(NW))

    J = 0
    g = [X[:, 0] - x0]
    for k in range(N):
        g.append(X[:, k + 1] - mdl.F(X[:, k], U[:, k]))
        du = U[:, k] - (u_prev if k == 0 else U[:, k - 1])
        J += r_p * du[0] ** 2 + r_y * du[1] ** 2
        xk = X[:, k + 1]
        rates = mdl.rates(xk)
        e_p = xk[0] - R[0, k]
        e_y = xk[3] - R[1, k]
        wk = terminal_weight if k == N - 1 else 1.0
        J += wk * (q_p * e_p ** 2 + q_y * e_y ** 2 + q_dp * rates[0] ** 2 + q_dy * rates[1] ** 2)

    nlp = {"x": ca.vertcat(ca.vec(X), ca.vec(U)), "f": J, "g": ca.vertcat(*g), "p": P}
    solver = ca.nlpsol("nmpc", "ipopt", nlp, json.loads(opts_json))
    nx_dec = NX * (N + 1)
    n_dec = nx_dec + NU * N
    lbx = np.full(n_dec, -np.inf)
    ubx = np.full(n_dec, np.inf)
    lbx[nx_dec:] = p.u_min
    ubx[nx_dec:] = p.u_max
    zeros_g = np.zeros(NX * (N + 1))
    return solver, nx_dec, lbx, ubx, zeros_g


def clear_solver_cache():
    _build_solver.cache_clear()


class NMPCController:
    name = "N-MPC"

    def __init__(self, p, Ts: float, N: int, weights: dict, terminal_weight: float = 5.0,
                 prediction_substeps: int = 2, reference_preview: bool = False,
                 ipopt_options: dict | None = None, store_predictions: bool = True):
        if not Ts > 0:
            raise ValueError(f"sample time Ts must be > 0 (got {Ts})")
        if int(N) < 1:
            raise ValueError(f"prediction horizon N must be >= 1 (got {N})")
        self.p, self.Ts, self.N = p, float(Ts), int(N)
        self.w = {k: float(weights[k]) for k in WEIGHT_NAMES}
        if not all(np.isfinite(v) and v >= 0 for v in self.w.values()):
            raise ValueError(f"N-MPC weights must be finite and >= 0: {self.w}")
        self.terminal_weight = float(terminal_weight)
        self.substeps = int(prediction_substeps)
        self.preview = bool(reference_preview)
        self.store_predictions = store_predictions

        opts = _solver_options(ipopt_options)
        self.warm_multipliers = opts.get("ipopt.warm_start_init_point") == "yes"
        (self.solver, self._nx_dec, self.lbx, self.ubx, self._zeros_g) = _build_solver(
            p, self.Ts, self.N, self.terminal_weight, self.substeps,
            json.dumps(opts, sort_keys=True))
        self._wvec = np.array([self.w[k] for k in WEIGHT_NAMES])
        self._horizon_t = self.Ts * np.arange(1, self.N + 1)
        self.reset(np.zeros(NX))

    # ------------------------------------------------------------------ API
    def reset(self, x0, n_steps: int | None = None):
        self.u_prev = np.zeros(NU)
        self.w0 = None
        self.lam_x0 = None
        self.lam_g0 = None
        self._n_pred = 0
        cap = int(n_steps) if n_steps else 0
        self._alloc_predictions(cap)

    def _alloc_predictions(self, cap: int):
        N = self.N
        if not self.store_predictions:
            cap = 0
        self._pred_t = np.empty(cap)
        self._pred_X = np.empty((cap, N + 1, NX))
        self._pred_U = np.empty((cap, N, NU))
        self._pred_R = np.empty((cap, N, NU))

    def _grow_predictions(self):
        """Amortised doubling, used only when reset() was not told the run length."""
        k = self._n_pred
        old = (self._pred_t, self._pred_X, self._pred_U, self._pred_R)
        self._alloc_predictions(max(64, 2 * len(old[0])))
        for new, o in zip((self._pred_t, self._pred_X, self._pred_U, self._pred_R), old):
            new[:k] = o[:k]

    def prediction_arrays(self) -> dict:
        """Recorded predictions as arrays: t (K,), X (K, N+1, 6), U (K, N, 2), R (K, N, 2)."""
        k = self._n_pred
        return {"t": self._pred_t[:k], "X": self._pred_X[:k], "U": self._pred_U[:k],
                "R": self._pred_R[:k]}

    @property
    def predictions(self) -> list:
        """Per-sample view (no copies) kept for the plotting/reporting interface."""
        a = self.prediction_arrays()
        return [{"t": a["t"][i], "X": a["X"][i], "U": a["U"][i], "R": a["R"][i]}
                for i in range(self._n_pred)]

    def _initial_guess(self, x):
        N = self.N
        if self.w0 is None:
            Xg = np.tile(np.asarray(x, float), N + 1)
            Ug = np.tile(self.u_prev, N)
            return np.concatenate([Xg, Ug])
        # shift the previous solution by one sample, repeating the last stage
        nxd = self._nx_dec
        w = np.empty_like(self.w0)
        w[:nxd - NX] = self.w0[NX:nxd]
        w[nxd - NX:nxd] = self.w0[nxd - NX:nxd]
        w[:NX] = x
        w[nxd:-NU] = self.w0[nxd + NU:]
        w[-NU:] = self.w0[-NU:]
        return w

    def _shifted_multipliers(self):
        nxd = self._nx_dec
        lg = np.empty_like(self.lam_g0)
        lg[:-NX] = self.lam_g0[NX:]
        lg[-NX:] = self.lam_g0[-NX:]
        lx = self.lam_x0.copy()
        lx[nxd:-NU] = self.lam_x0[nxd + NU:]
        lx[-NU:] = self.lam_x0[-NU:]
        return lx, lg

    def step(self, t, x, ref_now, ref_fn=None):
        N = self.N
        if self.preview and ref_fn is not None:
            R = ref_fn(t + self._horizon_t)                           # (2, N)
            r_flat = R.T.ravel()
        else:
            r_flat = np.tile(np.asarray(ref_now, float), N)          # [r_p, r_y] per stage
        pvec = np.concatenate([np.asarray(x, float), r_flat, self.u_prev, self._wvec])
        args = dict(x0=self._initial_guess(x), p=pvec, lbx=self.lbx, ubx=self.ubx,
                    lbg=self._zeros_g, ubg=self._zeros_g)
        if self.lam_g0 is not None:
            args["lam_x0"], args["lam_g0"] = self._shifted_multipliers()
        t0 = time.perf_counter()
        sol = self.solver(**args)
        solve_ms = (time.perf_counter() - t0) * 1e3
        stats = self.solver.stats()

        wopt = np.asarray(sol["x"]).ravel()
        nxd = self._nx_dec
        U = wopt[nxd:].reshape(N, NU)
        Xpred = wopt[:nxd].reshape(N + 1, NX)
        finite = bool(np.all(np.isfinite(wopt)))
        ok = bool(stats.get("success", False)) and finite
        if finite:
            u = np.clip(U[0], self.p.u_min, self.p.u_max)
            self.w0 = wopt
            if self.warm_multipliers:
                self.lam_x0 = np.asarray(sol["lam_x"]).ravel()
                self.lam_g0 = np.asarray(sol["lam_g"]).ravel()
        elif self.w0 is not None:
            # numerical failure: follow the previous plan one step further and shift it,
            # so repeated failures keep advancing along that plan instead of repeating it
            self.w0 = self._initial_guess(x)
            u = np.clip(self.w0[nxd:nxd + NU], self.p.u_min, self.p.u_max)
            self.lam_x0 = self.lam_g0 = None
        else:
            u = self.u_prev.copy()
        self.u_prev = u

        if self.store_predictions:
            if self._n_pred >= len(self._pred_t):
                self._grow_predictions()
            k = self._n_pred
            self._pred_t[k] = t
            self._pred_X[k] = Xpred
            self._pred_U[k] = U
            self._pred_R[k] = r_flat.reshape(N, NU)
            self._n_pred += 1

        log = {
            "compute_time_ms": solve_ms,
            "nmpc_solve_time_ms": solve_ms,
            "nmpc_ipopt_iterations": int(stats.get("iter_count", -1)),
            "nmpc_success": ok,
            "nmpc_return_status": str(stats.get("return_status", "")),
            "nmpc_cost": float(sol["f"]),
            "nmpc_pred_pitch_end": float(Xpred[-1, 0]),
            "nmpc_pred_yaw_end": float(Xpred[-1, 3]),
            "nmpc_pred_pitch_next": float(Xpred[1, 0]),
            "nmpc_pred_yaw_next": float(Xpred[1, 3]),
            "nmpc_constraint_violation": float(np.max(np.abs(np.asarray(sol["g"])))),
            "u_unsat_pitch": float(U[0, 0]),
            "u_unsat_yaw": float(U[0, 1]),
        }
        return u, log

    def describe(self) -> dict:
        return {"type": "N-MPC", "solver": "CasADi/IPOPT multiple shooting",
                "horizon_steps": self.N, "horizon_seconds": self.N * self.Ts,
                "weights": self.w, "terminal_weight": self.terminal_weight,
                "prediction_integrator": f"RK4 x{self.substeps} substeps",
                "reference_preview": self.preview, "Ts": self.Ts,
                "warm_start": "shifted primal + shifted multipliers"}
