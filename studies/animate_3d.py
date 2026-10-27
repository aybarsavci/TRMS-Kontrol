"""3D replay of a simulated TRMS run: PID and N-MPC side by side, with a guided tour.

Reads the telemetry written by ``python main.py`` and builds one self-contained
HTML page (three.js is embedded, so it also works offline):

* PID and N-MPC are shown side by side on one timeline, each with the same white
  target beam, so the difference is visible in a single frame. A view switch shows
  one controller alone (e.g. N-MPC only for a presentation that does not cover PID);
* the N-MPC's own predictions over its horizon (``nmpc_predictions.npz``) are drawn
  as fading future beams and a dotted path: what the controller "sees ahead";
* a guided tour splits the run into one chapter per reference step. Each chapter
  explains what changes and compares the two controllers with numbers measured
  from this run (time to settle within ±2°, overshoot, cross-coupling, oscillation,
  input saturation), and the replay can stop at the end of every chapter;
* both propellers spin from the simulated rotor speeds (drawn at a fraction of the
  real rpm, which a 60 Hz screen cannot show), with thrust arrows, a live
  comparison table, and a strip chart with the accumulated tracking error.

    python -m studies.animate_3d                       # latest run in outputs/
    python -m studies.animate_3d --run outputs/run_X   # a specific run
    python -m studies.animate_3d --open                # also open it in the browser

Output: <run>/trms_3d.html. A run made with ``--controllers nmpc`` gives an
N-MPC-only page; a run with both controllers gives the side-by-side comparison.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from trms.params import TRMSParams

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "templates" / "trms_3d.html"
THREE_JS = HERE / "vendor" / "three.min.js"

# telemetry column -> short key used by the page
COLUMNS = {"time": "t", "pitch": "pitch", "yaw": "yaw", "ref_pitch": "rp", "ref_yaw": "ry",
           "omega_main_rpm": "wm", "omega_tail_rpm": "wt", "thrust_main_N": "fm",
           "thrust_tail_N": "ft", "u_pitch": "uv", "u_yaw": "uh"}
CONTROLLER_ORDER = ("PID", "N-MPC")          # left to right in the side-by-side view
BAND_DEG = 2.0                                # "on target" band used throughout the page
HOLD_S = 1.0                                  # must stay inside the band this long before the step ends
PRED_STRIDE = 2                               # keep every 2nd predicted sample of the horizon
DEG = 180.0 / np.pi


def latest_run(root: Path) -> Path:
    runs = sorted(p for p in root.glob("run_*") if p.is_dir()
                  and any((p / c / "telemetry.csv").exists() for c in CONTROLLER_ORDER))
    if not runs:
        raise SystemExit(f"No run with telemetry found in {root}. Run `python main.py` first.")
    return runs[-1]


# ------------------------------------------------------------------ data
def load_telemetry(run: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for name in CONTROLLER_ORDER:
        f = run / name / "telemetry.csv"
        if f.exists():
            out[name] = pd.read_csv(f)
    if not out:
        raise SystemExit(f"{run} has no PID/ or N-MPC/ telemetry.csv")
    return out


def page_series(df: pd.DataFrame, Ts: float) -> dict:
    d = {COLUMNS[c]: np.round(df[c].to_numpy(float), 5).tolist() for c in COLUMNS}
    # accumulated tracking error (pitch + yaw), in degree-seconds so it reads in the same unit as the angles
    e = np.abs(df["ref_pitch"] - df["pitch"]) + np.abs(df["ref_yaw"] - df["yaw"])
    d["cum"] = np.round(np.r_[0.0, np.cumsum(e.to_numpy(float))[:-1]] * Ts * DEG, 3).tolist()
    return d


def load_predictions(run: Path) -> dict | None:
    f = run / "N-MPC" / "nmpc_predictions.npz"
    if not f.exists():
        return None
    z = np.load(f)
    X = z["X"][:, ::PRED_STRIDE, :]                      # (K, J, 6), j = 0 is "now"
    return {"J": int(X.shape[1]),
            "pitch": np.round(X[:, :, 0], 4).ravel().tolist(),
            "yaw": np.round(X[:, :, 3], 4).ravel().tolist()}


def scenario_events(run: Path) -> tuple[list, float]:
    cfg = yaml.safe_load((run / "settings_used.yaml").read_text(encoding="utf-8"))
    sc = cfg["scenarios"]["evaluation"]
    return [tuple(e) for e in sc["events"]], float(sc["duration"])


# ------------------------------------------------------------------ per-step facts
def step_facts(df: pd.DataFrame, axis: str, t0: float, t1: float, r_from: float, r_to: float) -> dict:
    """What one controller did during one reference step [t0, t1). Angles in degrees."""
    t = df["time"].to_numpy()
    m = (t >= t0 - 1e-9) & (t < t1 - 1e-9)
    other = "yaw" if axis == "pitch" else "pitch"
    tt = t[m]
    y = df[axis].to_numpy()[m] * DEG
    e = (df[f"ref_{axis}"].to_numpy()[m] - df[axis].to_numpy()[m]) * DEG
    eo = (df[f"ref_{other}"].to_numpy()[m] - df[other].to_numpy()[m]) * DEG
    inside = np.abs(e) < BAND_DEG
    stays = np.flip(np.cumprod(np.flip(inside))).astype(bool)   # inside from here to the end of the step
    settle = float(tt[np.argmax(stays)] - t0) if stays.any() else None
    if settle is not None and settle > (t1 - t0) - HOLD_S:     # only grazed the band at the very end
        settle = None
    d = (r_to - r_from) * DEG
    over = max(0.0, float(np.max(np.sign(d) * (y - r_to * DEG)))) / abs(d) * 100 if d else 0.0
    last = tt >= t1 - 2.0
    swing = float(np.ptp(y[last])) if last.any() else 0.0
    u_main = df["u_pitch"].to_numpy()[m]
    return {"settle_s": settle, "overshoot_pct": over, "max_err": float(np.abs(e).max()),
            "cross_max": float(np.abs(eo).max()), "swing_end": swing,
            "sat_main_pct": float(np.mean(np.abs(u_main) > 0.999) * 100),
            "iae": float(np.sum(np.abs(e) + np.abs(eo)) * (tt[1] - tt[0] if len(tt) > 1 else 0))}


def _num(x, nd=1):
    """Turkish decimal comma: 1.5 -> '1,5'."""
    return f"{x:.{nd}f}".replace(".", ",")


def _ctrl_line(name: str, f: dict) -> str:
    s = (f"{_num(f['settle_s'])} s'de ±{BAND_DEG:.0f}° bandına oturdu" if f["settle_s"] is not None
         else "adım bitene kadar ±2° bandına oturamadı")
    s += f", aşım %{f['overshoot_pct']:.0f}" if f["overshoot_pct"] >= 1 else ", aşım yok"
    if f["settle_s"] is None and f["swing_end"] > 2 * BAND_DEG:
        s += f", son 2 saniyede hâlâ {f['swing_end']:.0f}° aralığında sallanıyor"
    return f"{name}: {s}."


def build_chapters(dfs: dict, events: list, duration: float) -> list:
    names = list(dfs)
    both = len(names) == 2
    ref = next(iter(dfs.values()))
    chapters = []
    last_ref = {"pitch": float(ref["pitch"].iloc[0]), "yaw": float(ref["yaw"].iloc[0])}
    for i, (axis, t0, value) in enumerate(events):
        t1 = events[i + 1][1] if i + 1 < len(events) else duration
        r_from, r_to = last_ref[axis], float(value)
        last_ref[axis] = r_to
        facts = {n: step_facts(df, axis, t0, t1, r_from, r_to) for n, df in dfs.items()}
        dd = (r_to - r_from) * DEG
        if axis == "pitch":
            title = f"Pitch (yukarı-aşağı): {_num(r_from * DEG)}° → {_num(r_to * DEG)}°"
            what = ("Kiriş dinlenme konumundan kalkıyor. Ana rotor, yerçekimine karşı kirişin ucunu "
                    "kaldıracak itkiyi üretmeli." if i == 0 else
                    f"Hedef {abs(dd):.0f}° {'yukarıda' if dd > 0 else 'aşağıda'}. Ana rotor hızını "
                    f"{'artırıp kirişi kaldırmalı' if dd > 0 else 'azaltıp kirişi indirmeli'}, "
                    "hedefte de yerçekimini tam dengeleyen hızda durmalı.")
        else:
            title = f"Yaw (sağa-sola): {_num(r_from * DEG)}° → {_num(r_to * DEG)}°"
            what = (f"Kuyruk rotoru kirişi {abs(dd):.0f}° döndürmeli. Pitch hedefi değişmiyor, ama "
                    "dönerken oluşan tepki torku pitch'i de iter: buna çapraz etkileşim denir.")
        lines = [_ctrl_line(n, facts[n]) for n in reversed(names)]   # N-MPC first
        if axis == "yaw" or max(f["cross_max"] for f in facts.values()) > 3:
            other = "Pitch'teki" if axis == "yaw" else "Yaw'daki"
            lines.append(f"{other} en büyük sapma: " + ", ".join(
                f"{n} {facts[n]['cross_max']:.0f}°" for n in reversed(names)) + ".")

        # one takeaway, chosen from the measured facts
        notes = []
        if both:
            P, N = facts["PID"], facts["N-MPC"]
            if N["settle_s"] is not None and P["settle_s"] is None:
                key = "N-MPC hedefe oturuyor, PID bu adımda oturamıyor."
            elif N["settle_s"] is not None and P["settle_s"] is not None and N["settle_s"] < P["settle_s"] - 0.5:
                key = f"İkisi de oturuyor; N-MPC {_num(P['settle_s'] - N['settle_s'])} s daha erken."
            elif N["settle_s"] is None and P["settle_s"] is not None:
                key = "Bu adımda PID daha önce oturuyor."
            else:
                key = "İki kontrolcü bu adımda benzer davranıyor."
            if axis == "yaw" and P["cross_max"] > 2 * N["cross_max"] and P["cross_max"] > 10:
                notes.append(f"PID'de pitch {P['cross_max']:.0f}° sapıyor: PID'in iki döngüsü birbirinden habersiz, "
                             "kuyruk rotorunun pitch'i iteceğini bilemiyor. N-MPC bunu modelinden biliyor.")
            elif axis == "yaw" and N["cross_max"] > P["cross_max"] + 3:
                notes.append(f"Burada N-MPC pitch'i daha çok bozuyor ({N['cross_max']:.0f}° / {P['cross_max']:.0f}°), "
                             "çünkü yaw'ı çok daha hızlı çeviriyor. Yine de daha erken toparlanıyor.")
            if P["settle_s"] is None and P["swing_end"] > 10:
                notes.append("PID'in sabit kazançları bu açıda fazla agresif kalıyor: kiriş hedefin etrafında sönmeyen bir "
                             "salınıma giriyor. Rotor itkisi açıya göre değiştiği için tek bir kazanç seti her yerde uymuyor.")
            watch = ("Sağdaki (N-MPC) kirişin beyaz hedefe oturuşunu, soldakinin (PID) hedefin etrafında "
                     "sallanmasını karşılaştırın." if key.startswith("N-MPC hedefe") else
                     "İki kirişin beyaz hedefe ne kadar sürede oturduğunu ve diğer eksendeki sapmayı izleyin.")
        else:
            n = names[0]
            f = facts[n]
            key = (f"{n} {_num(f['settle_s'])} saniyede hedefe oturuyor." if f["settle_s"] is not None
                   else f"{n} bu adımda hedefe oturamıyor.")
            watch = "Kirişin beyaz hedefe yaklaşmasını ve hata çubuğunun yeşile dönmesini izleyin."
        if "N-MPC" in facts and facts["N-MPC"]["sat_main_pct"] > 10:
            notes.append(f"N-MPC'nin bedeli: ana motor bu adımda sürenin %{facts['N-MPC']['sat_main_pct']:.0f} kadarında "
                     "sınırda, yani tam güçte çalışıyor (alttaki tabloda SINIRDA uyarısı).")
        chapters.append({"t0": float(t0), "t1": float(t1), "axis": axis, "title": title, "what": what,
                         "lines": lines, "key": key, "notes": notes, "watch": watch, "facts": facts})
    return chapters


def build_summary(run: Path, series: dict, chapters: list) -> dict:
    rows = []
    for n in reversed(list(series)):                    # N-MPC first
        g = json.loads((run / n / "global_metrics.json").read_text(encoding="utf-8"))
        settled = [c["facts"][n]["settle_s"] for c in chapters]
        ok = [s for s in settled if s is not None]
        rows.append({"name": n,
                     "settled": f"{len(ok)} / {len(settled)}",
                     "settle_mean": f"{_num(np.mean(ok))} s" if ok else "–",
                     "iae": f"{series[n]['cum'][-1]:.0f}°·s",
                     "effort": _num(g["effort_int_u2_total"]),
                     "sat": f"%{g['saturation_fraction_pitch'] * 100:.0f}",
                     "cpu": f"{_num(g['compute_time_mean_ms'], 2)} ms"})
    text = []
    if len(rows) == 2:
        n, pd_ = rows[0], rows[1]
        cum = {r["name"]: series[r["name"]]["cum"][-1] for r in rows}
        g = {r["name"]: json.loads((run / r["name"] / "global_metrics.json").read_text(encoding="utf-8")) for r in rows}
        text.append(f"N-MPC {n['settled']} adımı ortalama {n['settle_mean']} içinde hedefe oturtuyor, "
                    f"PID ise {pd_['settled']} adımda oturabiliyor. Toplam takip hatası N-MPC'de "
                    f"{cum['PID'] / cum['N-MPC']:.0f} kat daha küçük.")
        text.append(f"N-MPC'nin bedeli: motorları daha sert kullanıyor (efor {n['effort']} / {pd_['effort']}, ana motor "
                    f"sürenin {n['sat']} kadarında sınırda) ve her 50 ms'de bir optimizasyon problemi çözüyor "
                    f"(ortalama {n['cpu']}, PID {pd_['cpu']}).")
        text.append("PID'in hesabı çok ucuz, ama sabit kazançlarla bu doğrusal olmayan ve eksenleri birbirine bağlı sistemde "
                    "yetersiz kalıyor: kazanç bir açıda iyiyse başka bir açıda salınım yapıyor ve iki döngü birbirini bozuyor.")
        text.append("Not: Bu bir simülasyon. N-MPC modeli birebir ve tüm durumları gürültüsüz biliyor; gerçek donanımda fark daralır.")
    return {"rows": rows, "text": text}


def concept_text(names: list, horizon_s: float | None) -> dict:
    c = {}
    if "PID" in names:
        c["PID"] = ("Her eksende ayrı bir döngü. Sadece o anki hataya (P), geçmişte biriken hataya (I) ve hatanın "
                    "değişim hızına (D) bakar. Sistemin modelini bilmez, geleceği tahmin etmez ve iki eksen "
                    "birbirinden habersizdir. Hesabı çok ucuzdur.")
    if "N-MPC" in names:
        c["N-MPC"] = (f"Her 50 ms'de sistemin doğrusal olmayan modeliyle {_num(horizon_s or 0)} saniye ileriyi "
                      "simüle eder ve hatayı en aza indiren motor komut dizisini optimizasyonla bulur. Bu dizinin sadece "
                      "ilk adımını uygular, 50 ms sonra her şeyi yeniden hesaplar. Turuncu kesikli yol ve soluk "
                      "kirişler, kontrolcünün o anki tahminidir.")
    return c


def build_page(run: Path) -> str:
    raw = load_telemetry(run)
    p = TRMSParams()
    any_df = next(iter(raw.values()))
    Ts = round(float(any_df["time"].iloc[1] - any_df["time"].iloc[0]), 6)
    events, duration = scenario_events(run)
    chapters = build_chapters(raw, events, duration)
    series = {n: page_series(df, Ts) for n, df in raw.items()}
    summary = build_summary(run, series, chapters)
    pred = load_predictions(run)
    horizon_s = None
    if "N-MPC" in raw:
        desc = json.loads((run / "N-MPC" / "controller_parameters.json").read_text(encoding="utf-8"))
        horizon_s = float(desc["controller"]["horizon_seconds"])
    for c in chapters:                                   # keep the page payload small
        c.pop("facts")
    data = {
        "Ts": Ts,
        "duration": duration - Ts,
        "controllers": series,
        "pred": pred,
        "predDt": Ts * PRED_STRIDE,
        "horizon_s": horizon_s,
        "band": BAND_DEG,
        "geometry": {"l_m": p.l_m, "l_t": p.l_t},
        "rpmMax": {"main": max(max(np.abs(s["wm"])) for s in series.values()) or 1.0,
                   "tail": max(max(np.abs(s["wt"])) for s in series.values()) or 1.0},
        "thrustMax": max(max(np.abs(np.r_[s["fm"], s["ft"]])) for s in series.values()) or 1.0,
        "chapters": chapters,
        "summary": summary,
        "concepts": concept_text(list(raw), horizon_s),
        "subtitle": f"{run.name} · Ts = {Ts * 1e3:.0f} ms · aynı model, aynı hedefler, aynı motor sınırları",
    }
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
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
