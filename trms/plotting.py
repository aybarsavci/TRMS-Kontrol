"""Matplotlib figures for single runs and the PID vs N-MPC comparison."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"PID": "#2a78d6", "N-MPC": "#eb6834"}
REF = "#52514e"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
EXTRA = ["#1baf7a", "#6250d6", "#e87ba4", "#008300"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 1.6,
    "legend.frameon": False, "legend.fontsize": 8,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _steps_vlines(ax, steps, axis=None):
    for s in steps:
        if axis is None or s["axis"] == axis:
            ax.axvline(s["t_start"], color=GRID, lw=1.0, zorder=0)


# ------------------------------------------------------------ single run
def plot_single_run(df, name, out: Path, steps, envelope=None, predictions=None):
    c = COLORS[name]
    t = df["time"].values

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, k, lab in ((axs[0], "pitch", "Pitch α_v"), (axs[1], "yaw", "Yaw α_h")):
        ax.plot(t, np.degrees(df[f"ref_{k}"]), "--", color=REF, lw=1.2, label="Reference")
        ax.plot(t, np.degrees(df[k]), color=c, label=name)
        _steps_vlines(ax, steps, k)
        ax.set_ylabel(f"{lab} [deg]")
        ax.legend(loc="best")
    axs[0].set_title(f"{name}: reference tracking")
    axs[1].set_xlabel("Time [s]")
    _save(fig, out / "01_tracking.png")

    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, k in zip(axs, ("pitch", "yaw")):
        ax.plot(t, np.degrees(df[f"err_{k}"]), color=c)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylabel(f"{k} error [deg]")
    axs[0].set_title(f"{name}: tracking error")
    axs[1].set_xlabel("Time [s]")
    _save(fig, out / "02_tracking_error.png")

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, k in zip(axs[:2], ("pitch", "yaw")):
        ax.plot(t, df[f"u_{k}"], color=c, label="applied u")
        if f"u_unsat_{k}" in df:
            ax.plot(t, df[f"u_unsat_{k}"].clip(-1.5, 1.5), color=c, lw=0.8, alpha=0.45,
                    label="controller output (pre-saturation, clipped at ±1.5)")
        ax.axhline(1, color=INK2, lw=0.8, ls=":")
        ax.axhline(-1, color=INK2, lw=0.8, ls=":")
        ax.set_ylabel(f"u_{k} [-]")
        ax.legend(loc="lower right")
    axs[2].plot(t, df["cum_int_u2_pitch"], color=c, label="∫u_pitch² dt")
    axs[2].plot(t, df["cum_int_u2_yaw"], color=c, ls="--", label="∫u_yaw² dt")
    axs[2].set_ylabel("Cumulative effort")
    axs[2].legend(loc="upper left")
    axs[0].set_title(f"{name}: control inputs and effort")
    axs[2].set_xlabel("Time [s]")
    _save(fig, out / "03_control_inputs.png")

    fig, axs = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    axs[0, 0].plot(t, df["omega_main_rpm"], color=c); axs[0, 0].set_ylabel("Main rotor [rpm]")
    axs[0, 1].plot(t, df["omega_tail_rpm"], color=c); axs[0, 1].set_ylabel("Tail rotor [rpm]")
    axs[1, 0].plot(t, df["thrust_main_N"], color=c); axs[1, 0].set_ylabel("Main thrust [N]")
    axs[1, 1].plot(t, df["thrust_tail_N"], color=c); axs[1, 1].set_ylabel("Tail thrust [N]")
    for a in axs[1]:
        a.set_xlabel("Time [s]")
    fig.suptitle(f"{name}: rotor speeds and thrusts", fontweight="bold")
    _save(fig, out / "04_rotors.png")

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for col, lab, cc in (("torque_aero_pitch", "aerodynamic", EXTRA[0]),
                         ("torque_gravity_pitch", "gravity", EXTRA[1]),
                         ("torque_centrifugal_pitch", "centrifugal", EXTRA[2]),
                         ("torque_friction_pitch", "friction", EXTRA[3])):
        axs[0].plot(t, df[col], color=cc, label=lab)
    axs[0].set_ylabel("Pitch torques [N·m]")
    axs[0].legend(ncol=4, loc="lower right")
    for col, lab, cc in (("torque_aero_yaw", "aerodynamic", EXTRA[0]),
                         ("torque_friction_yaw", "friction", EXTRA[3])):
        axs[1].plot(t, df[col], color=cc, label=lab)
    axs[1].set_ylabel("Yaw torques [N·m]")
    axs[1].legend(ncol=2, loc="lower right")
    axs[1].set_xlabel("Time [s]")
    axs[0].set_title(f"{name}: torque breakdown")
    _save(fig, out / "05_torques.png")

    fig, axs = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
    for ax, col, lab in zip(axs.ravel(),
                            ("pitch_rate", "yaw_rate", "x_S_v", "x_S_h", "x_u_vv", "x_u_hh"),
                            ("Ω_v pitch rate [rad/s]", "Ω_h yaw rate [rad/s]", "S_v [rad/s]",
                             "S_h [rad/s]", "u_vv main motor state", "u_hh tail motor state")):
        ax.plot(t, df[col], color=c)
        ax.set_ylabel(lab)
    for a in axs[2]:
        a.set_xlabel("Time [s]")
    fig.suptitle(f"{name}: full state trajectory", fontweight="bold")
    _save(fig, out / "06_states.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(np.degrees(df["yaw"]), np.degrees(df["pitch"]), color=c, lw=1.2, label=name)
    ax.plot(np.degrees(df["ref_yaw"]), np.degrees(df["ref_pitch"]), "o", color=REF, ms=5,
            label="Reference set-points")
    ax.set_xlabel("Yaw [deg]"); ax.set_ylabel("Pitch [deg]")
    ax.set_title(f"{name}: pitch-yaw path")
    ax.legend()
    _save(fig, out / "07_phase_pitch_yaw.png")

    # controller internals
    if name == "PID":
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for ax, k in zip(axs, ("pitch", "yaw")):
            for term, cc in (("P", EXTRA[0]), ("I", EXTRA[1]), ("D", EXTRA[2])):
                ax.plot(t, df[f"pid_{term}_{k}"], color=cc, label=term)
            ax.plot(t, df[f"u_{k}"], color=c, lw=1.0, label="u (applied)")
            ax.set_ylabel(f"{k} PID terms")
            ax.legend(ncol=4, loc="lower right")
        axs[0].set_title("PID: P / I / D contributions")
        axs[1].set_xlabel("Time [s]")
        _save(fig, out / "08_pid_terms.png")
    else:
        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axs[0].plot(t, df["nmpc_solve_time_ms"], color=c)
        axs[0].axhline(df["time"].diff().median() * 1e3, color=INK2, ls=":", lw=1,
                       label="sample time Ts")
        axs[0].set_ylabel("Solve time [ms]"); axs[0].legend()
        axs[1].plot(t, df["nmpc_ipopt_iterations"], color=c, drawstyle="steps-post")
        axs[1].set_ylabel("IPOPT iterations")
        axs[2].semilogy(t, df["nmpc_cost"].clip(lower=1e-12), color=c)
        axs[2].set_ylabel("Optimal cost")
        axs[2].set_xlabel("Time [s]")
        axs[0].set_title("N-MPC: solver statistics")
        _save(fig, out / "08_nmpc_solver.png")

        if predictions:
            fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            Ts = t[1] - t[0]
            stride = max(1, len(predictions) // 30)
            for ax, i, k in ((axs[0], 0, "pitch"), (axs[1], 3, "yaw")):
                ax.plot(t, np.degrees(df[f"ref_{k}"]), "--", color=REF, lw=1.0, label="Reference")
                for j, pr in enumerate(predictions[::stride]):
                    tp = pr["t"] + Ts * np.arange(pr["X"].shape[0])
                    ax.plot(tp, np.degrees(pr["X"][:, i]), color=c, lw=0.7, alpha=0.5,
                            label="Predicted horizons" if j == 0 else None)
                ax.plot(t, np.degrees(df[k]), color=INK, lw=1.0, label="Actual")
                ax.set_ylabel(f"{k} [deg]")
                ax.legend(loc="best")
            axs[0].set_title("N-MPC: open-loop predictions along the run")
            axs[1].set_xlabel("Time [s]")
            _save(fig, out / "09_nmpc_predictions.png")

    if envelope is not None:
        pitches, trims = envelope
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.degrees(pitches), trims, color=INK2)
        ax.set_xlabel("Pitch [deg]"); ax.set_ylabel("Trim main input u_v")
        ax.set_title("Steady-state input needed to hold pitch")
        _save(fig, out / "10_trim_envelope.png")


def plot_convergence(history, name, out: Path):
    if not history:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(history) + 1), history, "o-", color=COLORS[name], ms=4)
    ax.set_xlabel("Generation"); ax.set_ylabel("Best objective J")
    ax.set_title(f"{name}: auto-tuning convergence")
    _save(fig, out / "00_tuning_convergence.png")


# ------------------------------------------------------------ comparison
def plot_comparison(dfs: dict, step_df, glob_df, steps, out: Path):
    t = next(iter(dfs.values()))["time"].values

    fig, axs = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, k in zip(axs, ("pitch", "yaw")):
        ax.plot(t, np.degrees(next(iter(dfs.values()))[f"ref_{k}"]), "--", color=REF, lw=1.2,
                label="Reference")
        for name, df in dfs.items():
            ax.plot(df["time"], np.degrees(df[k]), color=COLORS[name], label=name)
        _steps_vlines(ax, steps, k)
        ax.set_ylabel(f"{k} [deg]")
        ax.legend(loc="best")
    axs[0].set_title("Reference tracking: PID vs N-MPC")
    axs[1].set_xlabel("Time [s]")
    _save(fig, out / "compare_01_tracking.png")

    fig, axs = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    for ax, k in zip(axs, ("pitch", "yaw")):
        for name, df in dfs.items():
            ax.plot(df["time"], np.degrees(df[f"err_{k}"]), color=COLORS[name], label=name)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylabel(f"{k} error [deg]")
        ax.legend(loc="best")
    axs[0].set_title("Tracking error")
    axs[1].set_xlabel("Time [s]")
    _save(fig, out / "compare_02_error.png")

    fig, axs = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for ax, k in zip(axs[:2], ("pitch", "yaw")):
        for name, df in dfs.items():
            ax.plot(df["time"], df[f"u_{k}"], color=COLORS[name], label=name)
        ax.set_ylabel(f"u_{k} [-]")
        ax.legend(loc="best")
    for name, df in dfs.items():
        axs[2].plot(df["time"], df["cum_int_u2_pitch"] + df["cum_int_u2_yaw"],
                    color=COLORS[name], label=name)
    axs[2].set_ylabel("∫(u_p²+u_y²) dt")
    axs[2].legend(loc="upper left")
    axs[0].set_title("Control effort")
    axs[2].set_xlabel("Time [s]")
    _save(fig, out / "compare_03_control_effort.png")

    fig, axs = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    for ax, k in zip(axs, ("pitch", "yaw")):
        for name, df in dfs.items():
            ax.plot(df["time"], df[f"cum_IAE_{k}"], color=COLORS[name], label=name)
        ax.set_ylabel(f"Cumulative IAE {k}")
        ax.legend(loc="upper left")
    axs[0].set_title("Accumulated absolute error")
    axs[1].set_xlabel("Time [s]")
    _save(fig, out / "compare_04_cumulative_iae.png")

    # per-step bar charts
    metrics = [("overshoot_pct", "Overshoot [%]"), ("settling_time_2pct_s", "Settling time 2% [s]"),
               ("rise_time_10_90_s", "Rise time 10-90% [s]"), ("IAE", "IAE per step [rad·s]")]
    labels = [f"{s['axis']} @{s['t_start']:.0f}s\n{np.degrees(s['r_from']):.0f}→{np.degrees(s['r_to']):.0f}°"
              for s in steps]
    names = list(dfs)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
    x = np.arange(len(steps))
    wbar = 0.38
    for ax, (m, lab) in zip(axs.ravel(), metrics):
        for j, name in enumerate(names):
            vals = step_df[step_df["controller"] == name].sort_values(["t_start", "axis"])[m].values
            pos = x + (j - 0.5) * wbar
            ax.bar(pos, np.nan_to_num(vals, nan=0.0), wbar * 0.92, color=COLORS[name], label=name)
            for xi, v in zip(pos, vals):
                if not np.isfinite(v):
                    ax.text(xi, 0, "not\nsettled", ha="center", va="bottom", fontsize=6, color=INK2)
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylabel(lab)
        ax.legend()
    fig.suptitle("Per-step performance", fontweight="bold")
    _save(fig, out / "compare_05_step_metrics.png")

    keys = [("IAE_pitch", "IAE pitch"), ("IAE_yaw", "IAE yaw"), ("RMSE_pitch", "RMSE pitch"),
            ("RMSE_yaw", "RMSE yaw"), ("effort_int_u2_total", "∫u² total"),
            ("total_variation_u_total", "TV(u) total")]
    fig, axs = plt.subplots(1, len(keys), figsize=(14, 3.4))
    for ax, (k, lab) in zip(axs, keys):
        vals = [float(glob_df.loc[n, k]) for n in names]
        ax.bar(names, vals, color=[COLORS[n] for n in names], width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_title(lab, fontsize=9)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Whole-run metrics (lower is better)", fontweight="bold")
    _save(fig, out / "compare_06_global_metrics.png")
