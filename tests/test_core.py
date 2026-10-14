"""Regression tests for the numerical core, the bug fixes and the optimisations.

Run with either:
    python -m pytest tests/          (if pytest is installed)
    python tests/test_core.py        (no extra dependency)
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from trms.cache import ParameterCache  # noqa: E402
from trms.controllers.nmpc import NMPCController  # noqa: E402
from trms.controllers.pid import PIDState, pid_step  # noqa: E402
from trms.metrics import step_metrics, tuning_objective  # noqa: E402
from trms.model import Plant, rest_state, rk4_np  # noqa: E402
from trms.params import TRMSParams  # noqa: E402
from trms.reporting import controller_metrics  # noqa: E402
from trms.scenario import Scenario  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
P = TRMSParams()


def test_casadi_plant_is_bit_identical_to_numpy_rk4():
    rng = np.random.default_rng(1)
    plant = Plant(P, 0.05, 10)
    for _ in range(20):
        x = rest_state(P) + rng.normal(0, 0.3, 6)
        u = rng.uniform(-1, 1, 2)
        ref = rk4_np(x[:, None], u[:, None], 0.05, P, 10)[:, 0]
        assert np.array_equal(plant.step(x, u), ref)
    X = rest_state(P)[:, None] + rng.normal(0, 0.3, (6, 64))
    Ub = rng.uniform(-1, 1, (2, 64))
    assert np.array_equal(plant.step(X, Ub), rk4_np(X, Ub, 0.05, P, 10))


def test_nan_candidate_is_penalised_not_rewarded():
    t = np.arange(100) * 0.05
    ref = np.zeros((100, 2))
    y = np.zeros((100, 2, 2))
    u = np.zeros((100, 2, 2))
    u[50, 0, 1] = np.nan                            # candidate 1 produced a NaN input
    steps = [dict(axis="pitch", axis_index=0, t_start=0.0, t_end=5.0,
                  r_from=-0.1, r_to=0.0, step_size=0.1)]
    J = tuning_objective(t, ref, y, u, steps, CFG["objective"], 0.05)
    assert J[0] == 0.0
    assert J[1] >= CFG["objective"]["divergence_penalty"]
    J2 = tuning_objective(t, ref, np.zeros((100, 2, 2)), np.zeros((100, 2, 2)), steps,
                          CFG["objective"], 0.05, diverged=np.array([False, True]))
    assert J2[1] >= CFG["objective"]["divergence_penalty"]


def test_metrics_survive_a_run_that_stopped_early():
    sc = Scenario.from_config("evaluation", CFG["scenarios"]["evaluation"], rest_state(P))
    n = 700                                          # telemetry ends at t = 35 s
    t = np.arange(n) * 0.05
    ref = sc.reference(t)
    import pandas as pd
    df = pd.DataFrame({"time": t, "ref_pitch": ref[0], "ref_yaw": ref[1],
                       "pitch": ref[0], "yaw": ref[1], "u_pitch": 0.5, "u_yaw": 0.0,
                       "compute_time_ms": 1.0})
    sdf, g = controller_metrics(df, sc.steps(), 0.05)
    assert len(sdf) == len(sc.steps())
    late = sdf[sdf["t_start"] >= 35.0]
    assert late["settled"].eq(False).all() and late["overshoot_pct"].isna().all()
    s_empty = dict(axis="pitch", t_start=100.0, t_end=110.0, r_from=0, r_to=1, step_size=1)
    assert step_metrics(t, ref[0], s_empty)["settled"] is False


def test_invalid_parameters_are_rejected():
    for bad in (dict(J_v=0.0), dict(T_mr=0.0), dict(F=-0.01, E=0.004), dict(u_min=1.0)):
        try:
            replace(P, **bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid parameters {bad}")


def test_rest_angle_handles_any_counterweight_sign():
    assert abs(P.rest_pitch - (-0.6711847037514624)) < 1e-15
    q = replace(P, C=0.0)                            # used to divide by zero
    assert np.isfinite(q.rest_pitch)


def test_pid_integrator_reaches_but_never_exceeds_the_limit():
    gains = np.array([[0.0], [5.0], [0.0], [0.0], [5.0], [0.0]])   # pure I, Ki*Ts*e = 2.5 per step
    st = PIDState.zeros(np.zeros((2, 1)))
    for _ in range(200):                             # large persistent error
        u, info = pid_step(gains, st, np.zeros((2, 1)), np.full((2, 1), 10.0), 0.05, 0.1, -1, 1)
    # old rule: the first step overshot the limit, so the integrator froze at 0 forever
    assert np.all(u == 1.0), u
    assert np.all(info["I"] <= 1.0)
    # error reverses: the integrator unwinds immediately (no windup reserve)
    u, info = pid_step(gains, st, np.zeros((2, 1)), np.full((2, 1), -0.1), 0.05, 0.1, -1, 1)
    assert np.all(u < 1.0)


def test_cache_write_is_atomic_and_roundtrips():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "c.json"
        c = ParameterCache(path)
        c.put("pid", "sig", {"Kp_pitch": 1.0}, {"objective": 2.0})
        assert ParameterCache(path).get("pid", "sig")["params"]["Kp_pitch"] == 1.0
        assert [f.name for f in Path(d).iterdir()] == ["c.json"]      # no temp files left
        assert json.loads(path.read_text(encoding="utf-8"))["pid"]["signature"] == "sig"


def test_nmpc_reuses_one_solver_per_horizon_and_shifts_warm_start():
    w = dict(q_pitch=1, q_yaw=1, q_rate_pitch=0.01, q_rate_yaw=0.01, r_pitch=0.01, r_yaw=0.01)
    a = NMPCController(P, 0.05, 6, w, ipopt_options=CFG["nmpc"]["ipopt"])
    b = NMPCController(P, 0.05, 6, {**w, "q_yaw": 7.0}, ipopt_options=CFG["nmpc"]["ipopt"])
    assert a.solver is b.solver                      # weights are parameters, not rebuilt
    x = rest_state(P)
    a.reset(x, n_steps=3)
    for k in range(3):
        u, log = a.step(k * 0.05, x, np.array([-0.3, 0.0]))
        assert log["nmpc_success"] and np.all(np.abs(u) <= 1)
    pa = a.prediction_arrays()
    assert pa["X"].shape == (3, 7, 6) and pa["U"].shape == (3, 6, 2)
    guess = a._initial_guess(x)
    nxd = a._nx_dec
    assert np.array_equal(guess[:6], x)
    assert np.array_equal(guess[6:nxd - 6], a.w0[12:nxd])           # states shifted by one
    assert np.array_equal(guess[nxd:-2], a.w0[nxd + 2:])            # inputs shifted by one
    lx, lg = a._shifted_multipliers()
    assert np.array_equal(lg[:-6], a.lam_g0[6:])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")
