import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({
    'font.size': 15,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'figure.dpi': 300
})

fig, ax = plt.subplots(figsize=(18, 10))

phases = {
    "Phase 1": {"y": 0.78, "color": "#FFB84D", "label": "DATA PREPARATION\n& INTEGRATION"},
    "Phase 2": {"y": 0.49, "color": "#4D94FF", "label": "PROBABILISTIC\nCLASSIFICATION\n& MODELING"},
    "Phase 3": {"y": 0.20, "color": "#66B266", "label": "UNCERTAINTY\nPROPAGATION\n& ANALYSIS"}
}

components = [
    {"name": "Building Geometry\nMaxar Precision3D\n3m CE90/LE90", "pos": (0.25, 0.78), "size": (0.14, 0.10), "color": "white", "phase": 1},
    {"name": "Zoning Records\nBarcelona Municipal\nKeyword Matching", "pos": (0.42, 0.78), "size": (0.14, 0.10), "color": "white", "phase": 1},
    {"name": "Feature Engineering\nPCA (95% variance)\n12 Geometric Features", "pos": (0.59, 0.78), "size": (0.15, 0.10), "color": "white", "phase": 1},
    {"name": "Material Intensities\nRASMI Database\nHeight-Based Assignment", "pos": (0.77, 0.78), "size": (0.17, 0.10), "color": "lightgray", "phase": 1},

    {"name": "AutoGluon Ensemble\n5-fold Cross-validation\nSMOTE Balancing", "pos": (0.30, 0.49), "size": (0.16, 0.12), "color": "lightgray", "phase": 2},
    {"name": "Probabilistic Classification\nSoft Typology Labels\nMultinomial Sampling", "pos": (0.51, 0.49), "size": (0.18, 0.12), "color": "lightgray", "phase": 2},
    {"name": "Monte Carlo Simulation\n3,000 Iterations\nJoint Uncertainty Modeling", "pos": (0.73, 0.49), "size": (0.18, 0.12), "color": "lightgray", "phase": 2},

    {"name": "Building-Level\nUncertainty\nDistributions", "pos": (0.30, 0.20), "size": (0.16, 0.12), "color": "white", "phase": 3},
    {"name": "District-Level\nSpatial Aggregation\n& Error Reduction", "pos": (0.51, 0.20), "size": (0.16, 0.12), "color": "white", "phase": 3},
    {"name": "Material-Specific\nUncertainty Attribution\nGeometric vs Typological", "pos": (0.72, 0.20), "size": (0.17, 0.12), "color": "white", "phase": 3},
]

phase_y_positions = [0.65, 0.36]
for y_pos in phase_y_positions:
    ax.plot([0.03, 0.86], [y_pos, y_pos], color='gray', linewidth=1.5, linestyle='-', alpha=0.7)

ax.plot([0.17, 0.17], [0.05, 0.85], color='gray', linewidth=1.5, linestyle='-', alpha=0.7)

for phase_name, phase_info in phases.items():
    ax.text(0.09, phase_info["y"], phase_name, 
           fontsize=18, fontweight='bold', ha='center', va='center',
           color='white',
           bbox=dict(boxstyle="round,pad=0.5", facecolor=phase_info["color"], 
                    edgecolor=phase_info["color"], linewidth=2.5))
    
    ax.text(0.09, phase_info["y"] - 0.06, phase_info["label"], 
           fontsize=12, fontweight='bold', ha='center', va='top',
           color=phase_info["color"])

for comp in components:
    rect = Rectangle((comp["pos"][0] - comp["size"][0]/2, comp["pos"][1] - comp["size"][1]/2),
                    comp["size"][0], comp["size"][1],
                    facecolor=comp["color"], edgecolor='black', linewidth=2,
                    alpha=0.9)
    ax.add_patch(rect)
    ax.text(comp["pos"][0], comp["pos"][1], comp["name"],
           ha='center', va='center', fontsize=13, fontweight='bold',
           wrap=True)

arrows = [
    {"start": (0.25, 0.73), "end": (0.25, 0.55), "style": "->", "color": "black", "width": 2},
    {"start": (0.36, 0.73), "end": (0.36, 0.55), "style": "->", "color": "black", "width": 2},
    {"start": (0.59, 0.73), "end": (0.36, 0.55), "style": "->", "color": "black", "width": 2},
    {"start": (0.76, 0.73), "end": (0.76, 0.55), "style": "->", "color": "black", "width": 2},

    {"start": (0.38, 0.49), "end": (0.42, 0.49), "style": "->", "color": "black", "width": 2},
    {"start": (0.60, 0.49), "end": (0.64, 0.49), "style": "->", "color": "black", "width": 2},

    {"start": (0.72, 0.43), "end": (0.30, 0.26), "style": "->", "color": "black", "width": 2},
    {"start": (0.72, 0.43), "end": (0.51, 0.26), "style": "->", "color": "black", "width": 2},
    {"start": (0.72, 0.43), "end": (0.72, 0.26), "style": "->", "color": "black", "width": 2},

    {"start": (0.32, 0.78), "end": (0.35, 0.78), "style": "->", "color": "black", "width": 2},  
    {"start": (0.49, 0.78), "end": (0.52, 0.78), "style": "->", "color": "black", "width": 2},  
]

for arrow in arrows:
    ax.annotate('', xy=arrow["end"], xytext=arrow["start"],
               arrowprops=dict(arrowstyle=arrow["style"], 
                             color=arrow["color"], 
                             lw=arrow["width"],
                             alpha=0.8))

ax.text(0.52, 0.06, 'Case Study: Sarrià-Sant Gervasi District, Barcelona\n1,746 buildings • Validation across mixed architectural periods and building types',
       ha='center', va='center', fontsize=14, style='italic',
       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

ax.set_xlim(0, 0.88)
ax.set_ylim(0, 1)
ax.axis('off')

formats = ['png']
for fmt in formats:
    plt.savefig(f"Figure_1.{fmt}", 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

plt.tight_layout()
plt.show()