import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np 

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 13
plt.rcParams["legend.title_fontsize"] = 14
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["figure.titleweight"] = "bold"

# Data (Panel a)
data = {
    "Material": ["Wood", "Brick", "Concrete", "Steel", "Glass", "Plastics", "Copper", "Aluminum"],
    "Building CV": [35.63, 33.19, 31.42, 28.52, 26.06, 23.58, 23.32, 23.29],
    "District CV": [2.03, 2.09, 1.40, 1.45, 1.37, 1.34, 1.33, 1.35],
    "Reduction Factor": [17.5, 15.9, 22.4, 19.7, 19.0, 17.6, 17.5, 17.3]
}
df = pd.DataFrame(data)

# Data for Panel (b)
df_stats = pd.DataFrame({
    'sample_size': [100, 200, 400, 800, 1600],
    'Concrete_cv_mean': [6.073154866, 4.158969517, 3.008381612, 2.171579939, 1.552391037],
    'Concrete_cv_std': [0.511074977, 0.491976477, 0.370398839, 0.394675536, 0.03863303],
    'Brick_cv_mean': [7.318074946, 5.74890265, 5.046303392, 3.410896563, 2.843282497],
    'Brick_cv_std': [0.543135732, 0.51987515, 0.523698606, 0.443177844, 0.072189545],
    'Wood_cv_mean': [7.865020589, 5.349672268, 4.215281865, 2.95745547, 2.203018234],
    'Wood_cv_std': [2.687215748, 1.338345011, 0.773046257, 0.512835033, 0.054588096],
    'Steel_cv_mean': [6.401236537, 4.331545777, 3.062225489, 2.245271947, 1.590041812],
    'Steel_cv_std': [0.59693704, 0.504920133, 0.354681864, 0.439708492, 0.038901222],
    'Glass_cv_mean': [6.030288378, 4.086984491, 2.908393828, 2.112762019, 1.509277818],
    'Glass_cv_std': [0.552576304, 0.488319333, 0.353354182, 0.420802066, 0.039096598],
    'Plastics_cv_mean': [5.882512367, 3.984253786, 2.851162537, 2.05884641, 1.482210681],
    'Plastics_cv_std': [0.591155478, 0.511557112, 0.366680345, 0.419325545, 0.041187035],
    'Aluminium_cv_mean': [5.953398784, 4.018601952, 2.853112251, 2.070443943, 1.485256521],
    'Aluminium_cv_std': [0.591885097, 0.495160092, 0.353654875, 0.426509424, 0.04041885],
    'Copper_cv_mean': [5.866754691, 3.964429836, 2.8195658, 2.041568492, 1.46705516],
    'Copper_cv_std': [0.598590081, 0.495144398, 0.352170032, 0.417333505, 0.040473162]
})

materials = ['Concrete', 'Brick', 'Wood', 'Steel', 'Glass', 'Plastics', 'Aluminium', 'Copper']
x_values = [100, 200, 400, 800, 1600]

# --- Layout ---
fig = plt.figure(figsize=(20, 18))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.4], hspace=0.25)

# Panel (a) 
ax1 = fig.add_subplot(gs[0])
x = range(len(df["Material"]))
building_color = "lightblue"
district_color = "#E69F00"

ax1.vlines(x, 0, df["Building CV"], color=building_color, linewidth=2)
ax1.scatter(x, df["Building CV"], color=building_color, edgecolor="black", s=120, label="Building CV (%)", zorder=3)

offset = 0.23
x_offset = [i + offset for i in x]
ax1.vlines(x_offset, 0, df["District CV"], color=district_color, linewidth=2)
ax1.scatter(x_offset, df["District CV"], color=district_color, edgecolor="black", marker="s", s=120, label="District CV (%)", zorder=3)

ax1.set_ylabel("Coefficient of Variation (%)")
ax1.set_xticks(x)
ax1.set_xticklabels(df["Material"], rotation=45)
ax1.set_ylim(-1, df["Building CV"].max() + 2.4)

for i, val in enumerate(df["Building CV"]):
    ax1.text(i, val + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=20)

for i, val in enumerate(df["District CV"]):
    ax1.text(x_offset[i], val + 0.5, f"{val:.2f}", ha="center", va="bottom", fontsize=20)

ax2 = ax1.twinx()
ax2.set_ylabel("Reduction Factor (×)")
ax2.set_ylim(ax1.get_ylim())

for i, val in enumerate(df["Reduction Factor"]):
    ax1.text(i, val, f"{val:.1f}x", fontsize=18, ha="center", va="center")

ax1.legend(loc="upper right", fontsize=16)
ax1.set_title("(a)", loc="left")

# Panel (c)
gs_c = gs[1].subgridspec(2, 4, wspace=0.25, hspace=0.25)
axes_c = [fig.add_subplot(gs_c[i, j]) for i in range(2) for j in range(4)]

ymins, ymaxs = [], []

for idx, (ax, material) in enumerate(zip(axes_c, materials)):
    y = df_stats[f'{material}_cv_mean']
    yerr = df_stats[f'{material}_cv_std']

    ax.plot(x_values, y, marker='o', linewidth=2)
    ax.fill_between(x_values, y - yerr, y + yerr, alpha=0.6)

    ax.set_title(material, fontsize=18, fontweight="bold")

    ymins.append((y - yerr).min())
    ymaxs.append((y + yerr).max())

    if idx >= 4:
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_values, rotation=90)
        ax.set_xlabel("Number of Samples")
    else:
        ax.set_xticks(x_values)
        ax.set_xticklabels([])

common_ymin = min(ymins)
common_ymax = max(ymaxs)
for ax in axes_c:
    ax.set_ylim(common_ymin, common_ymax)

fig.text(0.088, 0.30, ' Mean Coefficient of Variation (%)', va='center', rotation='vertical', fontsize=20)
axes_c[0].set_title("(b)", loc="left")

plt.tight_layout()

output_dir = "Output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "Figure_5.png"), dpi=300, bbox_inches="tight")
plt.show()
