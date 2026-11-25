"""Presentation charts. Run after the pipeline:  python src/charts.py"""
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
FIG.mkdir(exist_ok=True)
for f in (ROOT / "assets" / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))

INK, SUB, GRID, BG, ACC, MUTED = "#1D1D1F", "#6E6E73", "#E8E8ED", "#FFFFFF", "#0071E3", "#C7C7CC"
plt.rcParams.update({
    "font.family": "Inter", "font.size": 12, "text.color": INK, "axes.labelcolor": SUB,
    "axes.edgecolor": GRID, "xtick.color": SUB, "ytick.color": SUB, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG, "savefig.dpi": 200,
})
SKEL = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def title(fig, t, s):
    fig.text(0.035, 0.965, t, fontsize=17, fontweight=600, color=INK, va="top")
    fig.text(0.035, 0.965 - 0.36 / fig.get_figheight(), s, fontsize=11.5, color=SUB, va="top")


def finish(fig, path):
    fig.tight_layout(rect=[0.01, 0, 1, 1 - 0.95 / fig.get_figheight()])
    fig.savefig(path)
    plt.close(fig)


def innings(m):
    pts = m["innings"]["points"]
    b = [0] + [p["balls"] for p in pts]
    r = [0] + [p["runs"] for p in pts]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ph = m["innings"]["phases"]
    for p in ph:
        if p["name"] == "Acceleration":
            ax.axvspan(p["from"] - 1, p["to"], color=ACC, alpha=.07, lw=0)
            ax.text((p["from"] - 1 + p["to"]) / 2, max(r) * 1.02, f"{p['runs']} runs in {p['balls']} balls",
                    ha="center", fontsize=10.5, color=ACC, fontweight=600)
    ax.plot(b, r, color=INK, lw=2, solid_capstyle="round")
    ax.scatter(b[1:], r[1:], s=14, color=INK, zorder=3)
    for p in pts:
        if p.get("ball_runs") in (4, 6):
            ax.scatter(p["balls"], p["runs"], s=110 if p["ball_runs"] == 6 else 70, color=ACC, zorder=4,
                       edgecolor="white", linewidth=1.2)
            ax.annotate(str(p["ball_runs"]), (p["balls"], p["runs"]), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=9.5, color=ACC, fontweight=600)
    ax.set_xlim(0, max(b) + 1)
    ax.set_ylim(0, max(r) * 1.12)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Balls faced")
    ax.set_ylabel("Runs")
    title(fig, f"{ph[2]['runs']} runs in {ph[2]['balls']} balls turned the innings",
          "Runs against balls faced, read from the broadcast scoreboard · blue dots are boundaries confirmed ball by ball")
    finish(fig, FIG / "innings.png")


def phases(m):
    ph = m["innings"]["phases"]
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    y = np.arange(len(ph))[::-1]
    srs = [p["strike_rate"] for p in ph]
    ax.barh(y, srs, color=[ACC if p["name"] == "Acceleration" else MUTED for p in ph], height=.56)
    for yi, p in zip(y, ph):
        ax.text(p["strike_rate"] + 6, yi, f"{p['strike_rate']}  ·  {p['runs']} off {p['balls']}", va="center",
                fontsize=11, color=INK if p["name"] == "Acceleration" else SUB,
                fontweight=600 if p["name"] == "Acceleration" else 400)
    ax.set_yticks(y, [f"{p['name']}  (balls {p['from']}–{p['to']})" for p in ph], fontsize=11.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(srs) * 1.35)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlabel("Strike rate (runs per 100 balls)")
    title(fig, "Patient middle, explosive finish", "Strike rate in each phase of the innings, from the scoreboard timeline")
    finish(fig, FIG / "phases.png")


def pose_at(st, lo, hi):
    ts = np.array(st["skeleton"]["times"])
    F = np.array([[np.nan if v is None else v for v in f] for f in st["skeleton"]["frames"]]).reshape(len(ts), 17, 2)
    P = np.nanmedian(F[(ts >= lo) & (ts <= hi)], axis=0)
    return P - np.nanmean(P[[11, 12]], axis=0)


def strokes(m):
    st = m["strokes"]
    c = m["consistency"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 6.2))
    for ax, (lo, hi, name, col) in zip(axes, [(-.9, -.5, "Stance  ·  0.5–0.9 s before contact", MUTED),
                                             (-.03, .03, "Contact  ·  the frame of fastest hands", ACC)]):
        for s in st:
            k = pose_at(s, lo, hi)
            for p, q in SKEL:
                ax.plot([k[p, 0], k[q, 0]], [-k[p, 1], -k[q, 1]], color=col if col == ACC else "#8E8E93",
                        lw=1.6, alpha=.55, solid_capstyle="round")
            ax.scatter([k[0, 0]], [-k[0, 1]], s=22, color=col if col == ACC else "#8E8E93", alpha=.6, zorder=3)
        ax.set_xlim(-.6, .6); ax.set_ylim(-.82, .74)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(name, fontsize=11.5, color=SUB, loc="left")
    fig.text(.29, .035, f"Average difference between deliveries: {c['stance_spread']*100:.1f}% of his height", ha="center", fontsize=10.5, color=SUB)
    fig.text(.75, .035, f"{c['contact_spread']*100:.1f}% of his height", ha="center", fontsize=10.5, color=ACC, fontweight=600)
    title(fig, f"Same set-up every ball. The shot varies {c['ratio']}x more than the stance",
          f"All {len(st)} tracked strokes overlaid · 17-point pose estimated from broadcast frames, scaled to his height")
    fig.tight_layout(rect=[0.01, 0.07, 1, 1 - 0.95 / fig.get_figheight()])
    fig.savefig(FIG / "strokes.png"); plt.close(fig)


if __name__ == "__main__":
    m = json.load(open(OUT / "metrics.json"))
    innings(m)
    phases(m)
    strokes(m)
    print("figures written")
