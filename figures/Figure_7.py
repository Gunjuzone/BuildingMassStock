import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Arial"],
    "font.size": 24,
})

# Define the dataset
data = {
    "Material": ["Concrete", "Brick", "Wood", "Steel", "Glass", "Plastics", "Aluminium", "Copper"],
    "Full Model CV (%)": [31.42, 33.19, 35.63, 28.52, 26.06, 23.58, 23.29, 23.32],
    "No Typology CV (%)": [27.11, 25.37, 27.37, 25.70, 24.53, 23.29, 23.23, 23.20],
    "No Geometry CV (%)": [15.51, 28.60, 26.28, 12.20, 7.26, 3.80, 1.36, 2.66],
    "No Height CV (%)": [22.67, 26.99, 31.20, 19.96, 17.33, 15.50, 15.06, 15.27],
    "No Area CV (%)": [24.47, 28.03, 32.60, 21.82, 19.24, 17.12, 16.47, 16.78]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Extract materials and CV types
materials = df["Material"].values
cv_types = df.columns[1:]
n_materials = len(materials)
n_types = len(cv_types)

# Set up plot
fig, ax = plt.subplots(figsize=(16, 9))

# Bar width and positions
bar_width = 0.16
x = np.arange(n_materials)

# Colors and hatches
colors = [
    "#E69F00",  
    "#56B4E9",  
    "#009E73",  
    "#F0E442",  
    "#0072B2", 
    "#D55E00",  
    "#CC79A7"   
]
hatches = ['/', '\\', '|', '--', '++']

# Plot grouped bars
for i, cv_type in enumerate(cv_types):
    values = df[cv_type].values
    bars = ax.bar(x + i * bar_width, values, width=bar_width,
                  label=cv_type, color=colors[i % len(colors)],
                  edgecolor="black", hatch=hatches[i % len(hatches)])
    
    # Add annotations
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f"{height:.1f}", rotation=90, ha='center', va='bottom', fontsize=20)

# Customize axes
ax.set_xticks(x + bar_width * (n_types - 1) / 2)
ax.set_xticklabels(materials, rotation=45, ha="right")
ax.set_ylabel("Coefficient of Variation (%)")

ax.set_ylim(0, df.iloc[:, 1:].values.max() + 3.5)

ax.legend(
    #title="CV Type",
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    frameon=True,
    fontsize=15,
    title_fontsize=15
)

# Layout and save
plt.tight_layout()
plt.savefig("Figure_7.png", dpi=300)
plt.show()
