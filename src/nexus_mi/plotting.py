"""Manuscript figure generation from analyzed NEXUS-MI outputs.

All plotted values are read from ``nexus-mi analyze`` outputs. The plotting
layout follows the final manuscript figures; no publication accuracy or
communication values are embedded in this module.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.ticker import FormatStrFormatter

POLICIES = ["Ideal", "P1", "P2", "P3", "P4", "P5", "P6"]
PANELS = [
    ("BCICIV-2a", "SB-PH"),
    ("BCICIV-2a", "EIB-PH"),
    ("OpenBMI", "SB-PH"),
    ("OpenBMI", "EIB-PH"),
]
STYLE = {
    "Ideal": dict(color="black", marker="*", ms=5.8, mew=0.70, lw=0.85),
    "P1": dict(color="#4C78A8", marker="o", ms=5.8, mew=0.70, lw=0.85),
    "P2": dict(color="#F58518", marker="s", ms=5.8, mew=0.70, lw=0.85),
    "P3": dict(color="#54A24B", marker="D", ms=5.8, mew=0.70, lw=0.85),
    "P4": dict(color="#B279A2", marker="^", ms=5.8, mew=0.70, lw=0.85),
    "P5": dict(color="#E45756", marker="P", ms=5.8, mew=0.70, lw=0.85),
    "P6": dict(color="#72B7B2", marker="X", ms=5.8, mew=0.70, lw=0.85),
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required analysis table not found: {path}. Run `nexus-mi analyze` first.")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(v: Any) -> float:
    return float(v)


def _marker_legend(fig, policies: list[str], labels: list[str] | None = None, *, y: float, ncol: int) -> None:
    handles = []
    for policy in policies:
        style = STYLE[policy]
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="none",
                linewidth=0.0,
                color=style["color"],
                marker=style["marker"],
                markersize=5.8,
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                markeredgewidth=0.70,
            )
        )
    fig.legend(
        handles,
        labels or policies,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, y),
        handlelength=0.9,
        handletextpad=0.40,
        columnspacing=1.0 if ncol > 3 else 1.35,
    )


def figure2(analysis_dir: Path, out_dir: Path) -> Path:
    rows = _read(analysis_dir / "figure2_accuracy_by_calibration.csv")
    data = {(r["dataset"], r["regime"], r["policy"], int(r["k"])): _f(r["mean_accuracy_pct"]) for r in rows}
    ks = [15, 20, 30]
    panel_axis = {
        ("BCICIV-2a", "SB-PH"): ((58.8, 73.8), [60, 62, 64, 66, 68, 70, 72]),
        ("BCICIV-2a", "EIB-PH"): ((61.0, 74.9), [62, 64, 66, 68, 70, 72, 74]),
        ("OpenBMI", "SB-PH"): ((69.35, 75.40), [70, 71, 72, 73, 74, 75]),
        ("OpenBMI", "EIB-PH"): ((77.90, 83.45), [78, 79, 80, 81, 82, 83]),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.75), constrained_layout=False)
    for ax, (dataset, regime) in zip(axes.ravel(), PANELS):
        for policy in POLICIES:
            y = [data[(dataset, regime, policy, k)] for k in ks]
            s = STYLE[policy]
            ax.plot(
                ks,
                y,
                color=s["color"],
                linewidth=s["lw"],
                marker=s["marker"],
                markersize=s["ms"],
                markeredgecolor=s["color"],
                markeredgewidth=s["mew"],
                solid_capstyle="round",
                zorder=4,
            )
        n = 9 if dataset == "BCICIV-2a" else 54
        ax.set_title(f"{dataset}, {regime} (n={n})", pad=4)
        ax.set_xticks(ks)
        ax.set_xlim(14.2, 30.8)
        ylim, yticks = panel_axis[(dataset, regime)]
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.grid(True, linewidth=0.30, alpha=0.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_xlabel("Session-2 calibration trials per class")
    axes[1, 1].set_xlabel("Session-2 calibration trials per class")
    axes[0, 0].tick_params(axis="x", labelbottom=False)
    axes[0, 1].tick_params(axis="x", labelbottom=False)
    _marker_legend(fig, POLICIES, y=0.015, ncol=7)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.955, bottom=0.205, hspace=0.36, wspace=0.24)
    path = out_dir / "Figure_2.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return FancyArrowPatch(
        (xdescent, ydescent + height / 2.0),
        (xdescent + width, ydescent + height / 2.0),
        arrowstyle="->",
        mutation_scale=fontsize * 1.1,
        lw=0.9,
        color="black",
    )


def figure3(analysis_dir: Path, out_dir: Path) -> Path:
    rows = _read(analysis_dir / "table_s1_policy_operating_points.csv")
    data = {(r["dataset"], r["regime"], r["policy"]): r for r in rows}
    style3 = {
        "Ideal": dict(color="black", marker="*", ms=8.0, mew=0.75, alpha=1.00),
        "P1": dict(color="#4C78A8", marker="o", ms=4.9, mew=0.65, alpha=0.68),
        "P2": dict(color="#F58518", marker="s", ms=5.0, mew=0.65, alpha=0.68),
        "P3": dict(color="#54A24B", marker="D", ms=6.9, mew=0.90, alpha=1.00),
        "P4": dict(color="#B279A2", marker="^", ms=5.2, mew=0.65, alpha=0.68),
        "P5": dict(color="#E45756", marker="P", ms=7.1, mew=0.90, alpha=1.00),
        "P6": dict(color="#72B7B2", marker="X", ms=5.4, mew=0.70, alpha=0.68),
    }
    label_offset = {
        ("BCICIV-2a", "SB-PH"): (0, 25),
        ("BCICIV-2a", "EIB-PH"): (0, 25),
        ("OpenBMI", "SB-PH"): (0, 23),
        ("OpenBMI", "EIB-PH"): (0, 23),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.95), constrained_layout=False)
    for ax, (dataset, regime) in zip(axes.ravel(), PANELS):
        p3 = data[(dataset, regime, "P3")]
        p5 = data[(dataset, regime, "P5")]
        for p in ("P3", "P5"):
            r = data[(dataset, regime, p)]
            ax.errorbar(
                _f(r["s2c_mb"]),
                _f(r["mean_accuracy_pct"]),
                yerr=_f(r["sem_accuracy_pct"]),
                fmt="none",
                ecolor="0.42",
                elinewidth=0.72,
                capsize=2.4,
                capthick=0.72,
                alpha=0.48,
                zorder=1,
            )
        for p in ["P1", "P2", "P4", "P6", "Ideal", "P3", "P5"]:
            r = data[(dataset, regime, p)]
            s = style3[p]
            ax.plot(
                _f(r["s2c_mb"]),
                _f(r["mean_accuracy_pct"]),
                linestyle="none",
                marker=s["marker"],
                markersize=s["ms"],
                color=s["color"],
                markeredgecolor=s["color"],
                markeredgewidth=s["mew"],
                alpha=s["alpha"],
                zorder=7 if p in {"P3", "P5"} else 5 if p == "Ideal" else 4,
            )
        x3, y3 = _f(p3["s2c_mb"]), _f(p3["mean_accuracy_pct"])
        x5, y5 = _f(p5["s2c_mb"]), _f(p5["mean_accuracy_pct"])
        ax.annotate(
            "",
            xy=(x5, y5),
            xytext=(x3, y3),
            arrowprops=dict(arrowstyle="->", lw=0.95, color="0.12", shrinkA=6, shrinkB=6),
            zorder=5,
        )
        red = 100.0 * (x3 - x5) / x3
        midpoint = ((x3 + x5) / 2.0, (y3 + y5) / 2.0)
        ax.annotate(
            "S2C reduction:\n" + rf"${red:.2f}\%$",
            xy=midpoint,
            xytext=label_offset[(dataset, regime)],
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=6.2,
            linespacing=1.34,
            color="0.12",
            bbox=dict(boxstyle="round,pad=0.13", facecolor="white", edgecolor="0.72", linewidth=0.35, alpha=0.97),
            zorder=9,
        )
        ax.set_title(f"{dataset}, {regime}", pad=4)
        ax.grid(True, color="0.86", linewidth=0.32, alpha=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        xs = [_f(data[(dataset, regime, p)]["s2c_mb"]) for p in POLICIES]
        ys = [_f(data[(dataset, regime, p)]["mean_accuracy_pct"]) for p in POLICIES]
        principal_lower = [
            _f(data[(dataset, regime, p)]["mean_accuracy_pct"]) - _f(data[(dataset, regime, p)]["sem_accuracy_pct"])
            for p in ("P3", "P5")
        ]
        principal_upper = [
            _f(data[(dataset, regime, p)]["mean_accuracy_pct"]) + _f(data[(dataset, regime, p)]["sem_accuracy_pct"])
            for p in ("P3", "P5")
        ]
        x_pad = (max(xs) - min(xs)) * 0.18
        y_min = min(min(ys), min(principal_lower))
        y_max = max(max(ys), max(principal_upper))
        y_pad = max(0.28, (y_max - y_min) * 0.06)
        ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
    fig.supxlabel("Server-to-client backbone traffic (MB)", y=0.10, fontsize=8.8)
    fig.supylabel("Mean accuracy (%)", x=0.022, fontsize=8.8)
    handles = []
    for p in POLICIES:
        s = style3[p]
        handles.append(Line2D([0], [0], linestyle="none", marker=s["marker"], markersize=s["ms"], color=s["color"], markeredgecolor=s["color"], markeredgewidth=s["mew"], alpha=s["alpha"]))
    handles.append(FancyArrowPatch((0, 0), (1, 0), arrowstyle="->", mutation_scale=12, lw=0.9, color="black"))
    fig.legend(
        handles,
        POLICIES + ["P3 to P5"],
        loc="lower center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
        handletextpad=0.35,
        columnspacing=0.9,
        handler_map={FancyArrowPatch: HandlerPatch(patch_func=_legend_arrow)},
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.955, bottom=0.19, hspace=0.35, wspace=0.22)
    path = out_dir / "Figure_3.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure4(analysis_dir: Path, out_dir: Path) -> Path:
    rows = _read(analysis_dir / "table_s1_policy_operating_points.csv")
    data = {(r["dataset"], r["regime"], r["policy"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.45), constrained_layout=False)
    c2s_color = "#4C78A8"
    s2c_color = "#9ECAE1"
    x = np.arange(len(POLICIES))
    for ax, dataset in zip(axes, ("BCICIV-2a", "OpenBMI")):
        c2s = np.array([_f(data[(dataset, "EIB-PH", p)]["c2s_mb"]) for p in POLICIES])
        s2c = np.array([_f(data[(dataset, "EIB-PH", p)]["s2c_mb"]) for p in POLICIES])
        totals = c2s + s2c
        ax.bar(x, c2s, width=0.60, color=c2s_color, edgecolor="white", linewidth=0.35, zorder=3)
        ax.bar(x, s2c, width=0.60, bottom=c2s, color=s2c_color, edgecolor="white", linewidth=0.35, zorder=3)
        label_offset = 0.38 if dataset == "BCICIV-2a" else 1.7
        for xpos, total in zip(x, totals):
            ax.text(xpos, total + label_offset, f"{total:.1f}", ha="center", va="bottom", fontsize=6.8, color="0.16", clip_on=False)
        ax.set_title(dataset, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(POLICIES)
        ax.grid(axis="y", linewidth=0.30, alpha=0.20, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Model communication\ntraffic (MB)", multialignment="center", linespacing=0.95)
    axes[0].set_ylim(0, 20.8)
    axes[0].set_yticks([0, 5, 10, 15, 20])
    axes[1].set_ylim(0, 122.5)
    axes[1].set_yticks([0, 20, 40, 60, 80, 100, 120])
    axes[1].legend(
        handles=[Patch(facecolor=c2s_color, edgecolor="none", label="Client-to-server"), Patch(facecolor=s2c_color, edgecolor="none", label="Server-to-client")],
        loc="upper right",
        frameon=False,
        handlelength=1.35,
        handleheight=0.75,
        borderaxespad=0.25,
        labelspacing=0.25,
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.20, wspace=0.18)
    path = out_dir / "Figure_4.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure5(analysis_dir: Path, out_dir: Path) -> Path:
    rows = _read(analysis_dir / "figure2_accuracy_by_calibration.csv")
    data = {(r["dataset"], r["regime"], r["policy"], int(r["k"])): _f(r["mean_accuracy_pct"]) for r in rows}
    policies = ["Ideal", "P3", "P5"]
    ks = [15, 20, 30]
    panel_axis = {
        ("BCICIV-2a", "SB-PH"): ((63.9, 70.15), [64, 66, 68, 70], "%.0f"),
        ("BCICIV-2a", "EIB-PH"): ((63.9, 70.55), [64, 66, 68, 70], "%.0f"),
        ("OpenBMI", "SB-PH"): ((71.62, 73.38), [72.0, 72.5, 73.0], "%.1f"),
        ("OpenBMI", "EIB-PH"): ((79.96, 81.44), [80.00, 80.25, 80.50, 80.75, 81.00, 81.25], "%.2f"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.15), constrained_layout=False)
    for ax, (dataset, regime) in zip(axes.ravel(), PANELS):
        for policy in policies:
            s = STYLE[policy]
            ax.plot(
                ks,
                [data[(dataset, regime, policy, k)] for k in ks],
                color=s["color"],
                linewidth=s["lw"],
                marker=s["marker"],
                markersize=s["ms"],
                markerfacecolor=s["color"],
                markeredgecolor=s["color"],
                markeredgewidth=s["mew"],
                solid_capstyle="round",
                zorder=4,
            )
        ax.set_title(f"{dataset}, {regime}", pad=4)
        ax.set_xticks(ks)
        ax.set_xlim(14.2, 30.8)
        ylim, yticks, fmt = panel_axis[(dataset, regime)]
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
        ax.grid(True, linewidth=0.30, alpha=0.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_xlabel("Calibration trials per class")
    axes[1, 1].set_xlabel("Calibration trials per class")
    axes[0, 0].tick_params(axis="x", labelbottom=False)
    axes[0, 1].tick_params(axis="x", labelbottom=False)
    _marker_legend(fig, policies, labels=["Ideal-link reference", "P3", "P5"], y=0.016, ncol=3)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.955, bottom=0.215, hspace=0.33, wspace=0.23)
    path = out_dir / "Figure_5.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _interpolate(start: tuple[float, float], end: tuple[float, float], fraction: float) -> tuple[float, float]:
    return (start[0] + fraction * (end[0] - start[0]), start[1] + fraction * (end[1] - start[1]))


def _direction_arrow(ax, start, end, color, linewidth):
    a = _interpolate(start, end, 0.44)
    b = _interpolate(start, end, 0.61)
    ax.annotate(
        "",
        xy=b,
        xytext=a,
        arrowprops={"arrowstyle": "->", "color": color, "linewidth": linewidth, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 8.0},
        zorder=3,
    )


def figure6(analysis_dir: Path, out_dir: Path) -> Path:
    sens = _read(analysis_dir / "table_s6_sensitivity.csv")
    paired = _read(analysis_dir / "figure6_sensitivity_paired_statistics.csv")
    data = {(r["dataset"], r["severity"], r["policy"]): r for r in sens}
    stats = {(r["dataset"], r["severity"]): r for r in paired}
    profiles = ["Mild", "Default", "Severe"]
    policies = ["P3", "P5"]
    style = {
        "P3": {"color": "#54A24B", "marker": "D", "markersize": 5.8, "markeredgewidth": 0.70, "linewidth": 0.85},
        "P5": {"color": "#E45756", "marker": "P", "markersize": 5.8, "markeredgewidth": 0.70, "linewidth": 0.85},
    }
    offsets = {
        ("BCICIV-2a", "P5", "Mild"): (7, 0), ("BCICIV-2a", "P5", "Default"): (7, -2), ("BCICIV-2a", "P5", "Severe"): (7, -11),
        ("BCICIV-2a", "P3", "Mild"): (-7, -7), ("BCICIV-2a", "P3", "Default"): (7, 8), ("BCICIV-2a", "P3", "Severe"): (7, -4),
        ("OpenBMI", "P5", "Mild"): (7, -10), ("OpenBMI", "P5", "Default"): (7, 7), ("OpenBMI", "P5", "Severe"): (7, -10),
        ("OpenBMI", "P3", "Mild"): (-7, 0), ("OpenBMI", "P3", "Default"): (7, -7), ("OpenBMI", "P3", "Severe"): (7, 6),
    }
    fig = plt.figure(figsize=(7.2, 4.25), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=[2.65, 1.18], hspace=0.48, wspace=0.24)
    top_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    diff_axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    for ax, dataset in zip(top_axes, ("BCICIV-2a", "OpenBMI")):
        for policy in policies:
            s = style[policy]
            points = [(_f(data[(dataset, profile, policy)]["s2c_mb"]), _f(data[(dataset, profile, policy)]["mean_accuracy_pct"])) for profile in profiles]
            ax.plot([p[0] for p in points], [p[1] for p in points], linestyle="-", linewidth=s["linewidth"], color=s["color"], solid_capstyle="round", solid_joinstyle="round", zorder=2)
            for start, end in zip(points[:-1], points[1:]):
                _direction_arrow(ax, start, end, s["color"], s["linewidth"])
            for profile, (x, y) in zip(profiles, points):
                ax.plot(x, y, linestyle="none", marker=s["marker"], markersize=s["markersize"], color=s["color"], markerfacecolor=s["color"], markeredgecolor=s["color"], markeredgewidth=s["markeredgewidth"], zorder=5)
                if profile == "Default":
                    ax.plot(x, y, linestyle="none", marker="o", markersize=9.0, markerfacecolor="none", markeredgecolor="0.12", markeredgewidth=0.80, zorder=6)
                dx, dy = offsets[(dataset, policy, profile)]
                ha = "right" if (dataset, policy, profile) in {("BCICIV-2a", "P3", "Mild"), ("OpenBMI", "P3", "Mild")} else "left"
                ax.annotate(profile, xy=(x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=6.8, fontweight="semibold" if profile == "Default" else "normal", color="0.18", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.06, "alpha": 0.94}, clip_on=False, zorder=7)
        ax.set_title(f"{dataset}, EIB-PH", pad=4)
        ax.set_xlabel("Server-to-client communication (MB)")
        ax.grid(True, linewidth=0.30, alpha=0.20, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.7)
        if dataset == "BCICIV-2a":
            ax.set_ylabel("Mean accuracy (%)")
            ax.set_xlim(2.55, 7.55); ax.set_xticks([3, 4, 5, 6, 7]); ax.set_ylim(67.55, 70.35); ax.set_yticks([68, 69, 70])
        else:
            ax.set_xlim(14.2, 46.5); ax.set_xticks([15, 20, 25, 30, 35, 40]); ax.set_ylim(79.92, 81.38); ax.set_yticks([80.0, 80.4, 80.8, 81.2])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    for ax, dataset in zip(diff_axes, ("BCICIV-2a", "OpenBMI")):
        xs = [0, 1, 2]
        means = [_f(stats[(dataset, profile)]["mean_p5_minus_p3_pp"]) for profile in profiles]
        lows = [m - _f(stats[(dataset, profile)]["ci95_low_pp"]) for m, profile in zip(means, profiles)]
        highs = [_f(stats[(dataset, profile)]["ci95_high_pp"]) - m for m, profile in zip(means, profiles)]
        ax.axvspan(0.55, 1.45, color="0.94", zorder=0)
        ax.axhline(0.0, color="0.25", linewidth=0.65, linestyle=(0, (3, 2)), zorder=1)
        ax.errorbar(xs, means, yerr=[lows, highs], fmt="o", markersize=4.7, markerfacecolor=style["P5"]["color"], markeredgecolor=style["P5"]["color"], markeredgewidth=0.65, ecolor="0.34", elinewidth=0.85, capsize=2.8, capthick=0.85, zorder=3)
        ax.plot(1, means[1], linestyle="none", marker="o", markersize=7.5, markerfacecolor="none", markeredgecolor="0.12", markeredgewidth=0.80, zorder=4)
        ax.set_xlim(-0.45, 2.45)
        ax.set_xticks(xs, ["Mild", "Default\n(primary)", "Severe"])
        ax.get_xticklabels()[1].set_fontweight("semibold")
        ax.set_xlabel("Gateway-availability profile", labelpad=2)
        ax.grid(axis="y", linewidth=0.30, alpha=0.20, zorder=0)
        ax.tick_params(axis="x", pad=2)
        ax.text(0.220, 0.92, "P5 better", transform=ax.transAxes, ha="left", va="top", fontsize=6.2, fontstyle="italic", color="0.30", zorder=5)
        ax.text(0.025, 0.08, "P3 better", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2, fontstyle="italic", color="0.30", zorder=5)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(0.7)
        if dataset == "BCICIV-2a":
            ax.set_ylabel("P5 accuracy gain over P3\n(percentage points)", labelpad=5)
            ax.set_ylim(-3.35, 4.25); ax.set_yticks([-3, 0, 3])
        else:
            ax.set_ylim(-1.25, 1.82); ax.set_yticks([-1, 0, 1])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    handles = [Line2D([0], [0], linestyle="none", color=style[p]["color"], marker=style[p]["marker"], markersize=style[p]["markersize"], markerfacecolor=style[p]["color"], markeredgecolor=style[p]["color"], markeredgewidth=style[p]["markeredgewidth"]) for p in policies]
    fig.legend(handles, policies, loc="center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.390), handlelength=0.9, handletextpad=0.40, columnspacing=1.20)
    fig.subplots_adjust(left=0.090, right=0.985, top=0.935, bottom=0.135)
    path = out_dir / "Figure_6.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure7(analysis_dir: Path, out_dir: Path) -> Path:
    rows = _read(analysis_dir / "figure7_subject_reliability.csv")
    by = {(r["dataset"], r["policy"], r["subject"]): r for r in rows if r["regime"] == "EIB-PH" and r["policy"] in {"P3", "P5"}}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.65), constrained_layout=False, sharey=True)
    for ax, dataset in zip(axes, ("BCICIV-2a", "OpenBMI")):
        subjects = sorted({s for d, p, s in by if d == dataset and p == "P3"}, key=lambda x: int(x))
        subject_rows = []
        for subject in subjects:
            p3 = by[(dataset, "P3", subject)]
            p5 = by[(dataset, "P5", subject)]
            p3_change = _f(p3["difference_vs_ideal_pp"])
            p5_change = _f(p5["difference_vs_ideal_pp"])
            subject_rows.append((subject, p3_change, p5_change, p5_change - p3_change))
        subject_rows.sort(key=lambda row: (row[3], int(row[0])))
        x = np.arange(len(subject_rows))
        p3_vals = np.array([r[1] for r in subject_rows])
        p5_vals = np.array([r[2] for r in subject_rows])
        ax.vlines(x, np.minimum(p3_vals, p5_vals), np.maximum(p3_vals, p5_vals), color="0.58", linewidth=0.55, alpha=0.80, zorder=2)
        for policy, values in (("P3", p3_vals), ("P5", p5_vals)):
            s = STYLE[policy]
            ax.plot(x, values, linestyle="none", marker=s["marker"], markersize=s["ms"], color=s["color"], markerfacecolor=s["color"], markeredgecolor=s["color"], markeredgewidth=s["mew"], zorder=4)
        ax.axhline(0.0, color="black", linewidth=0.70, zorder=1)
        for threshold in (-5.0, 5.0):
            ax.axhline(threshold, color="0.25", linewidth=0.60, linestyle=(0, (1.2, 1.8)), zorder=1)
        ax.set_title(dataset, pad=4)
        ax.set_xlim(-0.55, len(subject_rows) - 0.45)
        ax.set_ylim(-15.5, 11.5)
        ax.set_yticks([-15, -10, -5, 0, 5, 10])
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(r[0])) for r in subject_rows], rotation=90)
        ax.grid(axis="y", linewidth=0.30, alpha=0.20, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=6.8 if dataset == "BCICIV-2a" else 4.9, pad=1.5 if dataset == "BCICIV-2a" else 1.0)
    axes[1].set_xlabel("Subject ID (sorted by P5-P3 change)")
    fig.supylabel("Accuracy change vs. ideal link (pp)", x=0.025, fontsize=8.5)
    _marker_legend(fig, ["P3", "P5"], y=0.008, ncol=2)
    fig.subplots_adjust(left=0.090, right=0.985, top=0.955, bottom=0.205, hspace=0.22)
    path = out_dir / "Figure_7.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all(analysis_dir: Path, out_dir: Path) -> list[Path]:
    analysis_dir = Path(analysis_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        figure2(analysis_dir, out_dir),
        figure3(analysis_dir, out_dir),
        figure4(analysis_dir, out_dir),
        figure5(analysis_dir, out_dir),
        figure6(analysis_dir, out_dir),
        figure7(analysis_dir, out_dir),
    ]
