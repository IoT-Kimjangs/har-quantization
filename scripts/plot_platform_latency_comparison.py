"""Plot default-thread median latency from the Windows and Mac reports."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/platform-latency-comparison.png"

DATA = {
    "WISDM": {
        "Keras predict": (208.7349, 24.409667),
        "Keras direct": (107.8605, 14.213708),
        "TFLite FP32": (0.0488, 0.044750),
        "FP16": (0.0399, 0.045125),
        "Dynamic INT8": (0.0333, 0.030875),
        "INT8 I/O, mixed FP32": (0.1921, 0.069291),
    },
    "PAMAP2": {
        "Keras predict": (213.0406, 25.687333),
        "Keras direct": (165.5518, 19.629521),
        "TFLite FP32": (0.5795, 0.255375),
        "FP16": (0.1566, 0.255916),
        "Dynamic INT8": (0.1132, 0.118417),
        "INT8 I/O, mixed FP32": (1.8892, 0.262875),
    },
}

plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "figure.dpi": 160})
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True, constrained_layout=True)
colors = {"Windows": "#2878B5", "Mac": "#E1812C"}
for ax, (dataset, values) in zip(axes, DATA.items()):
    names = list(values)
    x = np.arange(len(names))
    width = 0.36
    windows = [values[name][0] for name in names]
    mac = [values[name][1] for name in names]
    bars_win = ax.bar(x - width / 2, windows, width, color=colors["Windows"], label="Windows")
    bars_mac = ax.bar(x + width / 2, mac, width, color=colors["Mac"], label="Mac")
    ax.set_yscale("log")
    ax.set_xticks(x, names, rotation=24, ha="right")
    ax.set_title(dataset)
    ax.grid(axis="y", which="both", alpha=.25)
    ax.bar_label(bars_win, labels=[f"{v:g}" for v in windows], padding=2, fontsize=8, rotation=90)
    ax.bar_label(bars_mac, labels=[f"{v:g}" for v in mac], padding=2, fontsize=8, rotation=90)
    ax.legend(frameon=False)
axes[0].set_ylabel("Median latency (ms, logarithmic scale)")
fig.suptitle("Default-thread inference latency by runtime path", fontsize=16)
fig.text(.5, -.015,
         "Single-session observations. PAMAP2 used different input schedules: Windows resampled 494 windows (420 unique); Mac cycled all 494.",
         ha="center", fontsize=9)
fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
print(OUTPUT)
