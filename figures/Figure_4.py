import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Arial"],
    "font.size": 10,
})

data = {
    "Material": ["Wood", "Brick", "Concrete", "Steel", "Glass", "Plastics", "Copper", "Aluminium"],
    "Mean Uncertainty (%)": [35.63, 33.19, 31.42, 28.52, 26.06, 23.58, 23.32, 23.29]
}

df = pd.DataFrame(data)

df_sorted = df.sort_values(by="Mean Uncertainty (%)", ascending=True)

plt.figure(figsize=(8, 5))

plt.hlines(y=df_sorted["Material"], xmin=0, xmax=df_sorted["Mean Uncertainty (%)"],
           color="gray", alpha=0.9, linewidth=2)

plt.plot(df_sorted["Mean Uncertainty (%)"], df_sorted["Material"], "o", 
         markersize=8, color="gray")

plt.xlabel("Mean Uncertainty (%)", fontsize=14)
plt.ylabel("Material", fontsize=14)

plt.grid(axis="x", linestyle="--", alpha=0.4)

plt.xlim(0, 40)

for x, y in zip(df_sorted["Mean Uncertainty (%)"], df_sorted["Material"]):
    plt.text(x + 0.5, y, f"{x:.2f}%", va="center", fontsize=10)

output_dir = "Output"
os.makedirs(output_dir, exist_ok=True)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "Figure_4.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()