import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Global Font Settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Arial"],
    "font.size": 14,
    "axes.titlesize": 24,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 16,
    "legend.fontsize": 12
})

# Data for CV (%)
cv_data = {
    "Material": ["Concrete", "Brick", "Wood", "Steel", "Glass", "Plastics", "Aluminium", "Copper"],
    "Amenities": [32.02, 31.96, 33.63, 28.80, 26.14, 23.21, 22.90, 22.91],
    "Institutional": [31.85, 35.28, 38.96, 28.92, 26.25, 23.88, 23.59, 23.64],
    "Mixed-Use": [31.48, 33.71, 35.59, 28.55, 26.09, 23.59, 23.31, 23.33],
    "Residential": [29.86, 30.57, 34.70, 27.61, 25.62, 23.63, 23.38, 23.39]
}
cv_df = pd.DataFrame(cv_data)

patterns = {"Amenities": "/", "Institutional": "\\", "Mixed-Use": "x", "Residential": "."}
colors = {"Amenities": "#E69F00", "Institutional": "#56B4E9", "Mixed-Use": "#009E73", "Residential": "#F0E442"}

# Data for Material Stock Distribution (Million kg)
stock_data = {
    "Typology": ["Mixed-Use", "Residential", "Institutional", "Amenities"],
    "Concrete": [861.54, 243.48, 194.27, 188.40],
    "Brick": [198.74, 72.07, 21.00, 19.48],
    "Wood": [12.34, 3.34, 4.73, 4.57],
    "Steel": [44.91, 13.86, 7.70, 7.46],
    "Glass": [2.77, 0.83, 0.56, 0.54],
    "Plastics": [1.12, 0.33, 0.24, 0.24],
    "Aluminium": [0.45, 0.14, 0.09, 0.09],
    "Copper": [0.17, 0.05, 0.03, 0.03],
    "Buildings": [1090, 198, 216, 242]
}
stock_df = pd.DataFrame(stock_data)

big_materials = ["Concrete", "Brick", "Wood", "Steel"]
small_materials = ["Glass", "Plastics", "Aluminium", "Copper"]

material_colors = {
    "Concrete": "#a6cee3",
    "Brick": "#fb9a99",
    "Wood": "#b2df8a",
    "Steel": "#fdbf6f",
    "Glass": "#cab2d6",
    "Plastics": "#ffff99",
    "Aluminium": "#8dd3c7",
    "Copper": "#bebada"
}
material_hatches = {
    "Concrete": "//",
    "Brick": "\\\\",
    "Wood": "xx",
    "Steel": "..",
    "Glass": "--",
    "Plastics": "++",
    "Aluminium": "oo",
    "Copper": "**"
}

# Create figure with 2 subplots
fig, axes = plt.subplots(2, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [1, 1.2]})
plt.subplots_adjust(hspace=0.25)

# (a)
bar_width = 0.2
x = np.arange(len(cv_df["Material"]))
for i, btype in enumerate(patterns.keys()):
    values = cv_df[btype].values
    bars = axes[0].bar(
        x + i*bar_width,
        values,
        width=bar_width,
        label=btype,
        color=colors[btype],
        hatch=patterns[btype],
        edgecolor="black",
        linewidth=1.2
    )
    for bar, val in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=14, rotation=90, 
            color="black", fontweight="bold"
        )
axes[0].set_xticks(x + 1.5*bar_width)
axes[0].set_xticklabels(cv_df["Material"])
axes[0].set_ylabel("Coefficient of Variation (%)")
axes[0].set_title("(a)", loc="left")
axes[0].legend(loc="upper right")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# (b)
y_pos = np.arange(len(stock_df["Typology"]))
lefts = np.zeros(len(stock_df))
bar_segments = {mat: [] for mat in big_materials}

for mat in big_materials:
    bars = axes[1].barh(
        y_pos,
        stock_df[mat],
        left=lefts,
        label=mat,
        color=material_colors[mat],
        edgecolor="black",
        linewidth=1.0,
        hatch=material_hatches[mat]
    )
    bar_segments[mat] = bars
    lefts += stock_df[mat]

# Add labels
for i, row in stock_df.iterrows():
    concrete_end = row["Concrete"]
    brick_end = concrete_end + row["Brick"]
    wood_end = brick_end + row["Wood"]
    steel_end = wood_end + row["Steel"]
    
    for mat in big_materials:
        val = row[mat]
        bar = bar_segments[mat][i]
        xpos = bar.get_x() + bar.get_width()/2
        ypos = bar.get_y() + bar.get_height()/2

        if mat in ["Concrete", "Brick"]:
            axes[1].text(xpos, ypos, f"{val:.1f}",
                         ha="center", va="center",
                         fontsize=13, rotation=90, color="black",weight="bold",
                       bbox=dict(boxstyle="round,pad=0.15", facecolor="white", 
                               edgecolor="none", alpha=1.0))

        elif mat == "Wood":
            axes[1].annotate(
                f"{val:.1f}",
                xy=(bar.get_x() + bar.get_width() * 0.8, ypos),
                xytext=(steel_end + 15, ypos + 0.33),
                fontsize=14, rotation=90,
                arrowprops=dict(arrowstyle="-", lw=0.8, color="black"),
                va="center", ha="left"
            )

        elif mat == "Steel":
            axes[1].annotate(
                f"{val:.1f}",
                xy=(bar.get_x() + bar.get_width() * 0.7, ypos),
                xytext=(steel_end + 5, ypos - 0.22),
                fontsize=14, rotation=90,
                arrowprops=dict(arrowstyle="-", lw=0.8, color="black"),
                va="center", ha="left"
            )

# Totals & building counts
for i, row in stock_df.iterrows():
    total = row[big_materials].sum() + row[small_materials].sum()
    axes[1].text(total + 35, i + 0.1, f"{total:.1f} Mkg", va="center", fontsize=14, fontweight="bold")
   # axes[1].text(total + 35, i - 0.2, f"n={row['Buildings']}", va="center", fontsize=14)

axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(stock_df["Typology"])
axes[1].set_xlabel("Major Material Mass (Million kg)")
axes[1].set_xlim(0, stock_df[big_materials + small_materials].sum(axis=1).max() + 200)
axes[1].set_title("(b)", loc="left")

# Inset for small materials
axins = inset_axes(axes[1], width="50%", height="50%", loc="upper right", borderpad=1.5)

# Title for inset
axins.text(0.65, 1.02, "Minor Materials (Glass, Plastics, Aluminium, Copper)",
           transform=axins.transAxes, ha="center", va="top",
           fontsize=11, fontweight="bold", bbox=dict(boxstyle="round,pad=0.4",
           facecolor="white", edgecolor="black", linewidth=1.2))

lefts_zoom = np.zeros(len(stock_df))
bar_segments_zoom = {}

for mat in small_materials:
    bars = axins.barh(
        y_pos,
        stock_df[mat],
        left=lefts_zoom,
        color=material_colors[mat],
        edgecolor="black",
        linewidth=1.0,
        hatch=material_hatches[mat],
        label=mat
    )
    bar_segments_zoom[mat] = bars
    lefts_zoom += stock_df[mat]

# Label inset
for i, row in stock_df.iterrows():
    glass_end = row["Glass"]
    plastics_end = glass_end + row["Plastics"]
    aluminium_end = plastics_end + row["Aluminium"]
    copper_end = aluminium_end + row["Copper"]
    
    for mat in small_materials:
        val = row[mat]
        bar = bar_segments_zoom[mat][i]
        xpos = bar.get_x() + bar.get_width()/2
        ypos = bar.get_y() + bar.get_height()/2
        
        if mat in ["Glass", "Plastics"]:
            axins.text(xpos, ypos, f"{val:.2f}",
                       ha="center", va="center",
                       fontsize=12, rotation=90, color="black", weight="bold",  
                       bbox=dict(boxstyle="round,pad=0.15", facecolor="white", 
                               edgecolor="black", alpha=1.0))
        elif mat == "Aluminium":
            axins.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() * 0.2, ypos),
                xytext=(copper_end + 0.3, ypos + 0.60),
                fontsize=13, rotation=90,  
                arrowprops=dict(arrowstyle="-", lw=0.8, color="black"),
                va="center", ha="left"
            )
        elif mat == "Copper":
            axins.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() * 0.2, ypos),
                xytext=(copper_end + 0.15, ypos - 0.15),
                fontsize=13, rotation=90,  
                arrowprops=dict(arrowstyle="-", lw=0.8, color="black"),
                va="center", ha="left"
            )

axins.set_yticks(y_pos)
axins.set_yticklabels(stock_df["Typology"], fontsize=11.5)  
axins.set_xlabel("Minor Material Mass (Million kg)", fontsize=11.5) 
axins.tick_params(axis="x", labelsize=11.5)  
axins.set_xlim(0, 5)
axins.grid(True, axis="x", linestyle="--", alpha=0.7)
axins.spines["top"].set_visible(False)
axins.spines["right"].set_visible(False)

# Legends
handles_big, labels_big = axes[1].get_legend_handles_labels()
handles_small, labels_small = axins.get_legend_handles_labels()
all_handles = handles_big + handles_small
all_labels = labels_big + labels_small
axes[1].legend(all_handles, all_labels,
               loc="upper center", bbox_to_anchor=(0.5, -0.12),
               ncol=4, frameon=False)

# Save figure
output_dir = "Output"
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, "Figure_6.png"), dpi=300, bbox_inches="tight")
plt.show()