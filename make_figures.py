"""Generates the result charts used in the PPT and report from results CSVs."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
BLUE, ORANGE, RED, GREEN, GREY = "#0037A4", "#F28C28", "#D7263D", "#2E8B57", "#8A94A6"

d = pd.read_csv("results/vizwiz/results.csv")
before = pd.read_csv("results/vizwiz_results_before_handfilter.csv")
u = d[d.usable]

# 1. latency breakdown
cols = {"t_preprocess_ms": "Preprocess", "t_detection_ms": "YOLOv8n\ndetection", "t_depth_ms": "MiDaS\ndepth",
        "t_ocr_ms": "PP-OCRv4\ntext", "t_reasoning_ms": "Reasoning"}
med = u[list(cols)].median()
fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.bar(list(cols.values()), med.values, color=[GREY, BLUE, BLUE, ORANGE, GREEN])
for b, v in zip(bars, med.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:.0f} ms" if v >= 1 else "<1 ms", ha="center", fontsize=11)
ax.set_ylabel("Median time per frame (ms)")
ax.set_title(f"Per-module latency on CPU (n = {len(u)} frames)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/latency_breakdown.png", dpi=200); plt.close(fig)

# 2. outcome distribution
counts = {"Skipped\n(blurred/dark)": int((~d.usable).sum()),
          "Clear / no alert": int((u.level == "CLEAR").sum() - (u[u.level == "CLEAR"].message.notna()).sum()),
          "Sign read only": int((u[u.level == "CLEAR"].message.notna()).sum()),
          "MEDIUM alert": int((u.level == "MEDIUM").sum()),
          "HIGH alert": int((u.level == "HIGH").sum())}
fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.barh(list(counts), list(counts.values()), color=[GREY, GREEN, BLUE, ORANGE, RED])
for b, v in zip(bars, counts.values()):
    ax.text(v + 1.5, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=11)
ax.invert_yaxis(); ax.set_xlabel("Number of images")
ax.set_title(f"Pipeline outcome on {len(d)} VizWiz images")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/outcomes.png", dpi=200); plt.close(fig)

# 3. blur score histogram
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.hist(d.blur_score.clip(upper=600), bins=40, color=BLUE, alpha=0.85)
ax.axvline(30, color=RED, ls="--", lw=2); ax.text(45, ax.get_ylim()[1] * 0.92, "threshold = 30", color=RED, backgroundcolor="white")
ax.set_xlabel("Variance of Laplacian (clipped at 600)"); ax.set_ylabel("Images")
ax.set_title("Sharpness of VizWiz images (higher = sharper)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/blur_hist.png", dpi=200); plt.close(fig)

# 4. hand filter effect
ub = before[before.usable]
labels = ["HIGH alerts", "MEDIUM alerts", "Alerts led by\n'person'"]
vb = [int((ub.level == "HIGH").sum()), int((ub.level == "MEDIUM").sum()), int((ub.top_object == "person").sum())]
va = [int((u.level == "HIGH").sum()), int((u.level == "MEDIUM").sum()), int((u.top_object == "person").sum())]
fig, ax = plt.subplots(figsize=(8, 4.2))
x = range(len(labels)); w = 0.36
b1 = ax.bar([i - w / 2 for i in x], vb, w, label="Before hand filter", color=GREY)
b2 = ax.bar([i + w / 2 for i in x], va, w, label="After hand filter", color=BLUE)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, str(int(b.get_height())), ha="center")
ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.legend(frameon=False)
ax.set_title("Effect of the hand filter on false alerts (VizWiz)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/hand_filter.png", dpi=200); plt.close(fig)

summary = {
    "total_images": len(d), "usable": int(d.usable.sum()), "skipped_blur": int((d.skip_reason == "blurred").sum()),
    "skipped_dark": int((d.skip_reason == "too dark").sum()),
    "with_obstacle": int((u.n_obstacles > 0).sum()), "with_text": int((u.n_text > 0).sum()),
    "HIGH": int((u.level == "HIGH").sum()), "MEDIUM": int((u.level == "MEDIUM").sum()),
    "median_ms": {k: round(float(v), 1) for k, v in u[list(cols) + ["t_total_ms"]].median().items()},
    "mean_total_ms": round(float(u.t_total_ms.mean()), 1),
    "median_no_ocr_ms": round(float((u.t_total_ms - u.t_ocr_ms).median()), 1),
    "before_hand": vb, "after_hand": va,
}
import json; json.dump(summary, open("results/summary.json", "w"), indent=2); print(json.dumps(summary, indent=2))
