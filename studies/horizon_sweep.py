"""N-MPC prediction-horizon study: how far ahead does the controller need to look?

Runs the evaluation scenario with the tuned N-MPC weights at several horizons N
(everything else unchanged) and reports tracking quality against solve time.

    python -m studies.horizon_sweep                      # default horizons
    python -m studies.horizon_sweep --horizons 10 20 38  # custom list

Outputs (outputs/studies/horizon_sweep_<stamp>/):
    horizon_sweep.csv          one row per horizon, every reported metric
    horizon_sweep.md           the same as a readable table
    horizon_sweep_tracking.png pitch/yaw tracking for a few horizons
    horizon_sweep_tradeoff.png tracking error and solve time against N
"""
from __future__ import annotations

import argparse
import shutil

import numpy as np
import pandas as pd

from studies.common import REPORT_COLUMNS, ROOT, load_setup, markdown_table, output_dir, run_many
from trms.plotting import COLORS, EXTRA, INK2, REF, _save, plt

DEFAULT_HORIZONS = [10, 15, 20, 25, 30, 38, 50]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizons", type=int, nargs="+", default=None,
                    help="horizons N to test (samples of Ts); the tuned N is always included")
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (default: all cores)")
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--publish", action="store_true",
                    help="also copy the figures and table to docs/results/studies/")
    args = ap.parse_args()

    s = load_setup()
    cfg, p, nm = s["cfg"], s["p"], s["nmpc"]
    N_tuned = int(nm["horizon_steps"])
    horizons = sorted(set((args.horizons or DEFAULT_HORIZONS) + [N_tuned]))
    Ts = cfg["simulation"]["Ts"]
    print(f"Horizon sweep: N = {horizons} (tuned N = {N_tuned}, Ts = {Ts} s)")

    results = run_many([dict(p_model=p, p_plant=p, N=N, weights=nm["weights"], cfg=cfg)
                        for N in horizons], args.workers)

    rows = []
    for N, r in zip(horizons, results):
        g = r["metrics"]
        rows.append({"N": N, "horizon_s": N * Ts, "tuned": N == N_tuned,
                     **{k: g.get(k, np.nan) for k in REPORT_COLUMNS}})
        print(f"  N = {N:3d} ({N * Ts:4.2f} s): IAE = {g['IAE_total']:.3f}, "
              f"solve mean/max = {g['compute_time_mean_ms']:.1f}/{g['compute_time_max_ms']:.0f} ms"
              + ("  DIVERGED" if g["diverged"] else ""))
    table = pd.DataFrame(rows).set_index("N")

    out = output_dir("horizon_sweep", args.output_root)
    table.to_csv(out / "horizon_sweep.csv")
    show = table[["horizon_s", "tuned", "diverged", "IAE_pitch", "IAE_yaw", "max_overshoot_pct_pitch",
                  "steps_not_settled_pitch", "steps_not_settled_yaw",
                  "compute_time_mean_ms", "compute_time_max_ms", "realtime_budget_exceeded_fraction"]]
    show = show.rename(columns={
        "horizon_s": "horizon [s]", "IAE_pitch": "IAE pitch", "IAE_yaw": "IAE yaw",
        "max_overshoot_pct_pitch": "max OS pitch [%]",
        "steps_not_settled_pitch": "unsettled pitch", "steps_not_settled_yaw": "unsettled yaw",
        "compute_time_mean_ms": "solve mean [ms]", "compute_time_max_ms": "solve max [ms]",
        "realtime_budget_exceeded_fraction": "over budget"})
    (out / "horizon_sweep.md").write_text(
        "# N-MPC prediction-horizon sweep\n\n"
        f"Evaluation scenario, tuned weights fixed, only N changes. Ts = {Ts} s, "
        f"real-time budget = {Ts * 1e3:.0f} ms per sample. Tuned horizon: N = {N_tuned}.\n\n"
        + markdown_table(show, {"horizon [s]": ".2f", "over budget": ".1%",
                                "solve mean [ms]": ".1f", "solve max [ms]": ".0f"})
        + "\n\nSolve times are wall-clock and vary between runs and machines; "
          "the trajectories and metrics are deterministic.\n")

    # ---------------------------------------------------------- figures
    pick = sorted({horizons[0], N_tuned, horizons[-1]} | {h for h in horizons if h == 20})
    palette = {N: c for N, c in zip(pick, [EXTRA[1], EXTRA[0], COLORS["N-MPC"], EXTRA[2]])}
    palette[N_tuned] = COLORS["N-MPC"]
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, k in zip(axs, ("pitch", "yaw")):
        ts0 = results[horizons.index(N_tuned)]["timeseries"]
        ax.plot(ts0["time"], np.degrees(ts0[f"ref_{k}"]), "--", color=REF, lw=1.2, label="Reference")
        for N in pick:
            ts = results[horizons.index(N)]["timeseries"]
            lab = (f"N = {N} ({N * Ts:.2f} s)" + (" – tuned" if N == N_tuned else "")
                   + (" – diverged" if table.loc[N, "diverged"] else ""))
            ax.plot(ts["time"], np.degrees(ts[k]), color=palette[N], lw=1.8 if N == N_tuned else 1.2,
                    label=lab)
        # a diverged run can spin the beam through many turns: keep the axis on the
        # reference range so the stable runs stay readable (the CSV holds the full data)
        r = np.degrees(ts0[f"ref_{k}"].values)
        ax.set_ylim(min(r.min(), -5) - 45, max(r.max(), 5) + 45)
        ax.set_ylabel(f"{k} [deg]")
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5)
    axs[0].set_title("N-MPC reference tracking for different prediction horizons "
                     "(axes clipped to the reference range)")
    axs[1].set_xlabel("Time [s]")
    _save(fig, out / "horizon_sweep_tracking.png")

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ok = ~table["diverged"].astype(bool)
    ax1.plot(table.index[ok], table["IAE_pitch"][ok], "o-", color=COLORS["N-MPC"], label="IAE pitch")
    ax1.plot(table.index[ok], table["IAE_yaw"][ok], "s-", color=EXTRA[1], label="IAE yaw")
    for N in table.index[~ok]:
        ax1.axvspan(N - 1.5, N + 1.5, color="#f3c6c6", zorder=0)
        ax1.text(N, 0.97, "diverged", transform=ax1.get_xaxis_transform(), ha="center", va="top",
                 color="#a32020", fontsize=8, rotation=90)
    ax1.axvline(N_tuned, color=INK2, lw=0.8, ls=":")
    ax1.set_xlabel("Prediction horizon N [samples of Ts]")
    ax1.set_ylabel("IAE [rad·s]  (lower is better)")
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(table.index, table["compute_time_mean_ms"], "^--", color=EXTRA[0], label="solve time, mean")
    ax2.plot(table.index, table["compute_time_p95_ms"], "v:", color=EXTRA[0], label="solve time, p95")
    ax2.axhline(Ts * 1e3, color="#a32020", lw=0.8, ls="--")
    ax2.text(table.index[-1], Ts * 1e3, f"{Ts * 1e3:.0f} ms budget ", color="#a32020", va="bottom",
             ha="right", fontsize=8)
    ax2.set_ylabel("IPOPT solve time [ms]")
    ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    ax1.set_title(f"Horizon trade-off: tracking error vs computation (tuned N = {N_tuned})")
    _save(fig, out / "horizon_sweep_tradeoff.png")

    if args.publish:
        dst = ROOT / "docs" / "results" / "studies"
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("horizon_sweep.md", "horizon_sweep.csv", "horizon_sweep_tracking.png",
                  "horizon_sweep_tradeoff.png"):
            shutil.copy(out / f, dst / f)
    print(f"Outputs written to: {out}")


if __name__ == "__main__":
    main()
