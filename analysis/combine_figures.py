"""Combine the four 5-day analysis figures into one 2x2 PNG."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = ["fire_growth.png", "grid_impact.png",
        "saidi_voltage.png", "fire_perimeter_day5.png"]
TITLES = ["Fire growth", "Grid impact",
          "SAIDI & min voltage", "Burn perimeter (day 5)"]

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
for ax, fn, ti in zip(axes.ravel(), FIGS, TITLES):
    img = mpimg.imread(os.path.join(HERE, fn))
    ax.imshow(img)
    ax.set_title(ti, fontsize=12, fontweight="bold")
    ax.axis("off")
fig.suptitle("SoCal 5-Day Santa-Ana Wildfire on real SRTM terrain — "
             "fire growth, grid impact & reliability",
             fontsize=15, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(HERE, "five_day_combined.png")
fig.savefig(out, dpi=110, bbox_inches="tight")
print("wrote", out)
