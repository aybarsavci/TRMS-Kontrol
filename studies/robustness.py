"""N-MPC robustness to model mismatch.

In the main run the N-MPC predicts with exactly the same equations and parameters
as the simulated plant, which flatters it. Here the controller keeps the NOMINAL
model (and the tuned N, Q, R) while the simulated "true" TRMS gets perturbed
parameters, one physical effect at a time plus a combined worst case.

    python -m studies.robustness
    python -m studies.robustness --publish   # also copy results to docs/results/studies/

Outputs (outputs/studies/robustness_<stamp>/):
    robustness.csv            one row per plant case, every reported metric
    robustness.md             readable table with the perturbation of each case
    robustness_iae.png        tracking error per case (nominal model vs perturbed plant)
    robustness_tracking.png   pitch/yaw tracking for the nominal and the harder cases
"""
from __future__ import annotations

import argparse
import dataclasses
import shutil

import numpy as np
import pandas as pd

from studies.common import REPORT_COLUMNS, ROOT, load_setup, markdown_table, output_dir, run_many
from trms.params import TRMSParams
from trms.plotting import COLORS, EXTRA, INK2, REF, _save, plt


def _scaled(p: TRMSParams, **factors) -> TRMSParams:
    """Copy of ``p`` with each named field multiplied by its factor."""
    return dataclasses.replace(p, **{k: getattr(p, k) * f for k, f in factors.items()})


def cases(p: TRMSParams) -> list[tuple[str, str, TRMSParams]]:
    """(label, what changed, plant parameters). The N-MPC model always stays ``p``."""
    fh_paper = list(p.F_h_poly)
    fh_paper[-2] = 0.8080          # the paper's printed (suspected misprint) tail coefficient
    return [
        ("nominal", "plant = model", p),
        ("beam inertia +20%", "J_v ×1.2", _scaled(p, J_v=1.2)),
        ("beam inertia -20%", "J_v ×0.8", _scaled(p, J_v=0.8)),
        ("rotor thrust -15%", "S_f ×0.85 (both rotors weaker)", _scaled(p, S_f=0.85)),
        ("rotor thrust +15%", "S_f ×1.15 (both rotors stronger)", _scaled(p, S_f=1.15)),
        ("friction ×2", "K_v, K_h ×2", _scaled(p, K_v=2.0, K_h=2.0)),
        ("main motor 30% slower", "T_mr ×1.3", _scaled(p, T_mr=1.3)),
        ("counterweight +5%", "A ×1.05 (rest angle moves)", _scaled(p, A=1.05)),
        ("combined", "J_v ×1.2, S_f ×0.85, T_mr ×1.3, K_v,K_h ×2",
         _scaled(p, J_v=1.2, S_f=0.85, T_mr=1.3, K_v=2.0, K_h=2.0)),
        ("tail coeff. as printed", "F_h linear term 0.0801 → 0.8080 (paper value)",
         dataclasses.replace(p, F_h_poly=tuple(fh_paper))),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (default: all cores)")
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--publish", action="store_true",
                    help="also copy the figures and table to docs/results/studies/")
    args = ap.parse_args()

    s = load_setup()
    cfg, p, nm = s["cfg"], s["p"], s["nmpc"]
    N = int(nm["horizon_steps"])
    plant_cases = cases(p)
    print(f"Robustness: {len(plant_cases)} plant cases, N-MPC model nominal, N = {N}")
    results = run_many([dict(p_model=p, p_plant=pp, N=N, weights=nm["weights"], cfg=cfg)
                        for _, _, pp in plant_cases], args.workers)

    rows = []
    for (label, change, _), r in zip(plant_cases, results):
        g = r["metrics"]
        rows.append({"case": label, "change": change, **{k: g.get(k, np.nan) for k in REPORT_COLUMNS}})
        print(f"  {label:24s} IAE pitch/yaw = {g['IAE_pitch']:.2f}/{g['IAE_yaw']:.2f}, "
              f"unsettled = {g['steps_not_settled_pitch']}/{g['steps_not_settled_yaw']}"
              + ("  DIVERGED" if g["diverged"] else ""))
    table = pd.DataFrame(rows).set_index("case")
    nominal_iae = table.loc["nominal", "IAE_total"]
    table["IAE_total_vs_nominal"] = table["IAE_total"] / nominal_iae

    out = output_dir("robustness", args.output_root)
    table.to_csv(out / "robustness.csv")
    show = table[["change", "diverged", "IAE_pitch", "IAE_yaw", "IAE_total_vs_nominal",
                  "max_overshoot_pct_pitch", "steps_not_settled_pitch", "steps_not_settled_yaw",
                  "saturation_fraction_pitch"]].rename(columns={
        "IAE_pitch": "IAE pitch", "IAE_yaw": "IAE yaw", "IAE_total_vs_nominal": "IAE vs nominal",
        "max_overshoot_pct_pitch": "max OS pitch [%]", "steps_not_settled_pitch": "unsettled pitch",
        "steps_not_settled_yaw": "unsettled yaw", "saturation_fraction_pitch": "u_pitch saturated"})
    (out / "robustness.md").write_text(
        "# N-MPC robustness to model mismatch\n\n"
        f"The N-MPC keeps the nominal model and the tuned weights (N = {N}); only the simulated "
        "plant changes. Evaluation scenario, full-state feedback, no noise. "
        "`IAE vs nominal` = total IAE divided by the nominal case.\n\n"
        + markdown_table(show, {"IAE vs nominal": ".2f", "u_pitch saturated": ".1%"}) + "\n")

    # ---------------------------------------------------------- figures
    labels = list(table.index)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ok = ~table["diverged"].astype(bool).values
    ax.barh(y - 0.2, table["IAE_pitch"], 0.4, color=COLORS["N-MPC"], label="IAE pitch")
    ax.barh(y + 0.2, table["IAE_yaw"], 0.4, color=EXTRA[1], label="IAE yaw")
    ax.axvline(table.loc["nominal", "IAE_pitch"], color=INK2, lw=0.8, ls=":")
    ax.set_yticks(y, [lab + ("  (diverged)" if not k else "") for lab, k in zip(labels, ok)])
    ax.invert_yaxis()
    finite = np.concatenate([table["IAE_pitch"].values[ok], table["IAE_yaw"].values[ok]])
    xmax = 1.15 * np.nanmax(finite)
    ax.set_xlim(0, xmax)
    for i, (a, b) in enumerate(zip(table["IAE_pitch"], table["IAE_yaw"])):
        for v, dy in ((a, -0.2), (b, 0.2)):
            if v > xmax:
                ax.text(xmax, i + dy, f"{v:.0f} →", ha="right", va="center", fontsize=7, color="white")
    ax.set_xlabel("IAE [rad·s] over the 60 s evaluation scenario (lower is better)")
    ax.set_title("N-MPC with the nominal model, controlling a perturbed plant")
    ax.legend(loc="lower right")
    _save(fig, out / "robustness_iae.png")

    order = np.argsort(table["IAE_total"].values)
    show_cases = ["nominal"] + [labels[i] for i in order[::-1] if labels[i] != "nominal"
                                and ok[i]][:3]
    pal = [COLORS["N-MPC"], EXTRA[1], EXTRA[0], EXTRA[2]]
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, k in zip(axs, ("pitch", "yaw")):
        ts0 = results[0]["timeseries"]
        ax.plot(ts0["time"], np.degrees(ts0[f"ref_{k}"]), "--", color=REF, lw=1.2, label="Reference")
        for c, lab in zip(pal, show_cases):
            ts = results[labels.index(lab)]["timeseries"]
            ax.plot(ts["time"], np.degrees(ts[k]), color=c, lw=1.8 if lab == "nominal" else 1.2, label=lab)
        ax.set_ylabel(f"{k} [deg]")
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5)
    axs[0].set_title("Tracking with model mismatch: nominal plant and the three hardest stable cases")
    axs[1].set_xlabel("Time [s]")
    _save(fig, out / "robustness_tracking.png")

    if args.publish:
        dst = ROOT / "docs" / "results" / "studies"
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("robustness.md", "robustness.csv", "robustness_iae.png", "robustness_tracking.png"):
            shutil.copy(out / f, dst / f)
    print(f"Outputs written to: {out}")


if __name__ == "__main__":
    main()
