"""3D animation of a simulated TRMS run: beam motion and spinning rotors.

Reads the telemetry written by ``python main.py`` and builds one self-contained
HTML page (three.js is embedded, so it also works offline) that replays the run
in 3D:

* the beam pitches and yaws exactly as simulated, with a translucent "ghost"
  showing the reference attitude;
* both propellers spin from the simulated rotor speeds (rpm), including the tail
  rotor reversing direction. Real speeds (up to ~5,700 rpm main, ~14,000 rpm tail)
  are far above what a 60 Hz screen can show, so blades are drawn at a selectable
  fraction of the real speed; the real rpm is shown next to the scene and a
  translucent disc grows with speed;
* thrust arrows, input bars with a saturation flag, and a pitch/yaw strip chart
  with a playhead (click to seek).

    python -m studies.animate_3d                       # latest run in outputs/
    python -m studies.animate_3d --run outputs/run_X   # a specific run
    python -m studies.animate_3d --open                # also open it in the browser

Output: <run>/trms_3d.html (one page; if the run has both controllers there is a
PID / N-MPC switch).
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from trms.params import TRMSParams

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "templates" / "trms_3d.html"
THREE_JS = HERE / "vendor" / "three.min.js"

# telemetry column -> short key used by the page
COLUMNS = {"time": "t", "pitch": "pitch", "yaw": "yaw", "ref_pitch": "rp", "ref_yaw": "ry",
           "omega_main_rpm": "wm", "omega_tail_rpm": "wt", "thrust_main_N": "fm",
           "thrust_tail_N": "ft", "u_pitch": "uv", "u_yaw": "uh"}
CONTROLLER_ORDER = ("N-MPC", "PID")


def latest_run(root: Path) -> Path:
    runs = sorted(p for p in root.glob("run_*") if p.is_dir()
                  and any((p / c / "telemetry.csv").exists() for c in CONTROLLER_ORDER))
    if not runs:
        raise SystemExit(f"No run with telemetry found in {root}. Run `python main.py` first.")
    return runs[-1]


def load_controllers(run: Path) -> dict:
    out = {}
    for name in CONTROLLER_ORDER:
        f = run / name / "telemetry.csv"
        if f.exists():
            df = pd.read_csv(f, usecols=list(COLUMNS))
            out[name] = {COLUMNS[c]: np.round(df[c].to_numpy(float), 5).tolist() for c in COLUMNS}
    if not out:
        raise SystemExit(f"{run} has no PID/ or N-MPC/ telemetry.csv")
    return out


def _limits(ctrls: dict, key: str, ref_key: str) -> list:
    """Strip-chart range [deg]: references plus the stable part of the motion, rounded to 10 deg."""
    vals = np.concatenate([np.r_[c[ref_key], c[key]] for c in ctrls.values()]) * 180 / np.pi
    refs = np.concatenate([c[ref_key] for c in ctrls.values()]) * 180 / np.pi
    lo, hi = np.percentile(vals, [1, 99])
    lo, hi = max(lo, refs.min() - 40), min(hi, refs.max() + 40)     # a diverged run must not flatten the chart
    return [float(np.floor(lo / 10) * 10), float(np.ceil(hi / 10) * 10)]


def build_page(run: Path) -> str:
    ctrls = load_controllers(run)
    p = TRMSParams()
    any_c = next(iter(ctrls.values()))
    Ts = round(any_c["t"][1] - any_c["t"][0], 6)
    duration = max(len(c["t"]) for c in ctrls.values()) * Ts - Ts
    settings = run / "settings_used.yaml"
    if settings.exists():
        import yaml
        duration = float(yaml.safe_load(settings.read_text(encoding="utf-8"))["scenarios"]["evaluation"]["duration"]) - Ts
    data = {
        "Ts": Ts,
        "duration": duration,
        "controllers": ctrls,
        "geometry": {"l_m": p.l_m, "l_t": p.l_t},
        "rpmMax": {"main": max(max(np.abs(c["wm"])) for c in ctrls.values()) or 1.0,
                   "tail": max(max(np.abs(c["wt"])) for c in ctrls.values()) or 1.0},
        "thrustMax": max(max(np.abs(c["fm"] + c["ft"])) for c in ctrls.values()) or 1.0,
        "limits": {"pitch": _limits(ctrls, "pitch", "rp"), "yaw": _limits(ctrls, "yaw", "ry")},
        "subtitle": f"{' / '.join(ctrls)} · {run.name} · Ts = {Ts * 1e3:.0f} ms",
        "note": ("Kiriş açıları ve rotor hızları simülasyon telemetrisinden birebir alınmıştır. "
                 "Pervaneler, ekranda görülebilmesi için seçilen oranda yavaşlatılarak çizilir; "
                 "gerçek hız rpm olarak yanda yazar. Turuncu: ana rotor (pitch), mavi: kuyruk rotoru (yaw). "
                 "Gri çizgi referans açıyı gösterir. Model: Feedback TRMS 33-220, arXiv:1306.4883."),
    }
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__THREE_JS__*/", THREE_JS.read_text(encoding="utf-8"), 1)
    return html.replace("/*__TRMS_DATA__*/null", blob, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="run folder (default: latest in outputs/)")
    ap.add_argument("--out", default=None, help="output file (default: <run>/trms_3d.html)")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    ap.add_argument("--fragment", action="store_true",
                    help="write the page without the <!doctype>/<html> wrapper (for hosts that add their own)")
    args = ap.parse_args()

    run = Path(args.run) if args.run else latest_run(ROOT / "outputs")
    page = build_page(run)
    if not args.fragment:
        page = ("<!doctype html>\n<html lang=\"tr\">\n<head>\n<meta charset=\"utf-8\">\n"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                + page.replace("<style>", "<style>\n[hidden]{display:none!important}", 1)
                .replace("\n<div class=\"app\">", "\n</head>\n<body>\n<div class=\"app\">", 1)
                + "\n</body>\n</html>\n")
    out = Path(args.out) if args.out else run / "trms_3d.html"
    out.write_text(page, encoding="utf-8")
    print(f"3D animation written to: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
