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
"""
from __future__ import annotations

import time

import numpy as np

from ..model import NU, NX, rk4_casadi

WEIGHT_NAMES = ("q_pitch", "q_yaw", "q_rate_pitch", "q_rate_yaw", "r_pitch", "r_yaw")


class NMPCController:
    name = "N-MPC"

    def __init__(self, p, Ts: float, N: int, weights: dict, terminal_weight: float = 5.0,
                 prediction_substeps: int = 2, reference_preview: bool = False,
                 ipopt_options: dict | None = None, store_predictions: bool = True):
        import casadi as ca

        self.p, self.Ts, self.N = p, float(Ts), int(N)
        self.w = {k: float(weights[k]) for k in WEIGHT_NAMES}
        self.terminal_weight = float(terminal_weight)
        self.substeps = int(prediction_substeps)
        self.preview = bool(reference_preview)
        self.store_predictions = store_predictions

        mdl = rk4_casadi(p, self.Ts, self.substeps)
        N = self.N
        X = ca.SX.sym("X", NX, N + 1)
        U = ca.SX.sym("U", NU, N)
        P = ca.SX.sym("P", NX + NU * N + NU)
        x0 = P[:NX]
        R = ca.reshape(P[NX:NX + NU * N], NU, N)
        u_prev = P[NX + NU * N:]

        w = self.w
        J = 0
        g = [X[:, 0] - x0]
        for k in range(N):
            g.append(X[:, k + 1] - mdl.F(X[:, k], U[:, k]))
            du = U[:, k] - (u_prev if k == 0 else U[:, k - 1])
            J += w["r_pitch"] * du[0] ** 2 + w["r_yaw"] * du[1] ** 2
            xk = X[:, k + 1]
            rates = mdl.rates(xk)
            e_p = xk[0] - R[0, k]
            e_y = xk[3] - R[1, k]
            wk = self.terminal_weight if k == N - 1 else 1.0
            J += wk * (w["q_pitch"] * e_p ** 2 + w["q_yaw"] * e_y ** 2
                       + w["q_rate_pitch"] * rates[0] ** 2 + w["q_rate_yaw"] * rates[1] ** 2)

        self._nx_dec = NX * (N + 1)
        nlp = {"x": ca.vertcat(ca.vec(X), ca.vec(U)), "f": J, "g": ca.vertcat(*g), "p": P}
        opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes",
                "ipopt.max_iter": 100, "ipopt.tol": 1e-6, "ipopt.acceptable_tol": 1e-4,
                "expand": True}
        for k, v in (ipopt_options or {}).items():
            opts[k if "." in k or k in ("print_time", "expand") else f"ipopt.{k}"] = v
        self.warm_multipliers = opts.get("ipopt.warm_start_init_point") == "yes"
        self.solver = ca.nlpsol("nmpc", "ipopt", nlp, opts)

        n_dec = self._nx_dec + NU * N
        self.lbx = np.full(n_dec, -np.inf)
        self.ubx = np.full(n_dec, np.inf)
        self.lbx[self._nx_dec:] = p.u_min
        self.ubx[self._nx_dec:] = p.u_max
        self.lbg = np.zeros(NX * (N + 1))
        self.ubg = np.zeros(NX * (N + 1))
        self.reset(np.zeros(NX))

    # ------------------------------------------------------------------ API
    def reset(self, x0):
        self.u_prev = np.zeros(NU)
        self.w0 = None
        self.lam_x0 = None
        self.lam_g0 = None
        self.predictions = []

    def _initial_guess(self, x):
        N = self.N
        if self.w0 is None:
            Xg = np.tile(np.asarray(x, float).reshape(NX, 1), (1, N + 1))
            Ug = np.tile(self.u_prev.reshape(NU, 1), (1, N))
            return np.concatenate([Xg.T.ravel(), Ug.T.ravel()])
        # shift previous solution by one sample
        Xp = self.w0[:self._nx_dec].reshape(N + 1, NX)
        Up = self.w0[self._nx_dec:].reshape(N, NU)
        Xg = np.vstack([Xp[1:], Xp[-1:]])
        Xg[0] = x
        Ug = np.vstack([Up[1:], Up[-1:]])
        return np.concatenate([Xg.ravel(), Ug.ravel()])

    def step(self, t, x, ref_now, ref_fn=None):
        N = self.N
        if self.preview and ref_fn is not None:
            R = ref_fn(t + self.Ts * np.arange(1, N + 1))           # (2, N)
        else:
            R = np.tile(np.asarray(ref_now, float).reshape(2, 1), (1, N))
        pvec = np.concatenate([np.asarray(x, float), R.T.ravel(), self.u_prev])
        args = dict(x0=self._initial_guess(x), p=pvec, lbx=self.lbx, ubx=self.ubx,
                    lbg=self.lbg, ubg=self.ubg)
        if self.lam_g0 is not None:
            args["lam_x0"], args["lam_g0"] = self.lam_x0, self.lam_g0
        t0 = time.perf_counter()
        sol = self.solver(**args)
        solve_ms = (time.perf_counter() - t0) * 1e3
        stats = self.solver.stats()

        wopt = np.asarray(sol["x"]).ravel()
        U = wopt[self._nx_dec:].reshape(N, NU)
        Xpred = wopt[:self._nx_dec].reshape(N + 1, NX)
        ok = bool(stats.get("success", False)) and np.all(np.isfinite(wopt))
        if np.all(np.isfinite(U[0])):
            u = np.clip(U[0], self.p.u_min, self.p.u_max)
            self.w0 = wopt
            if self.warm_multipliers:
                self.lam_x0 = np.asarray(sol["lam_x"]).ravel()
                self.lam_g0 = np.asarray(sol["lam_g"]).ravel()
        else:  # numerical failure: fall back to the shifted previous plan
            u = self.u_prev.copy() if self.w0 is None else \
                self.w0[self._nx_dec:].reshape(N, NU)[min(1, N - 1)]
        self.u_prev = u

        if self.store_predictions:
            self.predictions.append(dict(t=t, X=Xpred.copy(), U=U.copy(), R=R.T.copy()))

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
                "reference_preview": self.preview, "Ts": self.Ts}
