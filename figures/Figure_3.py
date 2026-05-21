import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Arial"],
    "font.size": 18,
})

feature_data = pd.DataFrame({
    "Feature": [
        "Height", "Form_ratio", "Y_coord", "Rectangularity", "Convexity",
        "X_coord", "Match_Score", "Area", "Aspect_Ratio", "Elongation",
        "Confidence_Score", "AMSL", "Solidity", "MBG_Width"
    ],
    "Selection_Count": [5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 3, 3, 2],
    "Avg_Importance": [
        1.8420649, 1.740704194, 1.699237046, 1.684518107, 1.655103544,
        1.466170008, 1.432528243, 1.328591536, 1.324978394, 1.324977966,
        1.353060106, 1.318943099, 1.312773338, 1.264904604
    ]
})

feature_data = feature_data.sort_values("Avg_Importance", ascending=True)

classes = ["Amenities", "Institutional", "Mixed-Use", "Residential"]
cm = np.array([
    [66.5, 12.8,  3.7, 16.9],
    [14.8, 71.8,  0.0, 13.4],
    [0.5,  0.3, 98.9,  0.4],
    [12.1,  7.6,  1.5, 78.8]
])

report_data = pd.DataFrame({
    "precision": [0.71, 0.76, 0.95, 0.69],
    "recall":    [0.67, 0.72, 0.99, 0.79],
    "f1-score":  [0.69, 0.75, 0.97, 0.74]
}, index=classes)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.3, wspace=0.3)

ax_main = fig.add_subplot(gs[0, :])

bars = ax_main.barh(feature_data["Feature"], feature_data["Selection_Count"],
                    color="lightblue", edgecolor="black", alpha=0.8, label="Selection Count")

ax_imp = ax_main.twiny()

line = ax_imp.plot(feature_data["Avg_Importance"], range(len(feature_data["Feature"])),
                   color="black", marker="o", linewidth=2.5, markersize=5, label="Avg Importance")

importance_values = feature_data["Avg_Importance"].values
y_positions = np.arange(len(feature_data))

for i, (imp, y_pos) in enumerate(zip(importance_values, y_positions)):
    base_offset = 0.04
    
    nearby_values = importance_values[max(0, i-2):min(len(importance_values), i+3)]
    if len(nearby_values) > 1:
        value_range = np.max(nearby_values) - np.min(nearby_values)
        if value_range < 0.1:
            offset_multiplier = 1 + (i % 3) * 0.8
            base_offset *= offset_multiplier
    
    ax_imp.annotate(f"{imp:.2f}", 
                    xy=(imp, y_pos), 
                    xytext=(imp + base_offset, y_pos),
                    va="center", ha="left", fontsize=16,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                    weight="regular")

ax_main.set_xlabel("Selection Count", fontsize=18)
ax_imp.set_xlabel("Avg Importance", fontsize=18)
ax_main.set_title("(a)", loc="left", fontsize=20, y=1.12)

ax_imp.set_xlim(1.2, 2.0)
ax_main.set_xlim(0, 5.2)

ax_main.grid(axis='x', alpha=0.3)
ax_imp.grid(axis='x', alpha=0.3)

legend_elements = [
    Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='black', alpha=0.8, label='Selection Count (bars)'),
    Line2D([0], [0], color='black', marker='o', linewidth=2.5, markersize=5, label='Avg Importance (line)')
]
ax_main.legend(handles=legend_elements, loc='lower right', fontsize=16, framealpha=0.9)

ax_cm = fig.add_subplot(gs[1, 0])
sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", cbar=True,
            xticklabels=classes, yticklabels=classes, ax=ax_cm,
            annot_kws={"size": 22, "weight": "regular"})
ax_cm.set_title("(b)", loc="left", fontsize=22)
ax_cm.set_xlabel("Predicted (%)", fontsize=20)
ax_cm.set_ylabel("Actual (%)", fontsize=20)

ax_cm.set_xticklabels(ax_cm.get_xticklabels(), rotation=45, ha="right")
ax_cm.set_yticklabels(ax_cm.get_yticklabels(), rotation=0)

ax_cr = fig.add_subplot(gs[1, 1])
sns.heatmap(report_data, annot=True, fmt=".2f", cmap="YlOrRd", cbar=True,
            ax=ax_cr, annot_kws={"size": 22, "weight": "regular"})
ax_cr.set_title("(c)", loc="right", fontsize=22)
ax_cr.set_xlabel("Metrics", fontsize=20)
ax_cr.set_ylabel("Typology", fontsize=20)

ax_cr.set_xticklabels(ax_cr.get_xticklabels(), rotation=45, ha="right")
ax_cr.set_yticklabels(ax_cr.get_yticklabels(), rotation=0)

plt.tight_layout()

output_dir = "Output"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, "Figure_3.png"),
            dpi=300, bbox_inches="tight", facecolor='white')
plt.show()