"""Can building morphology predict cadastral targets instead of zoning designations?"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

from train_classifier import LOCATION, MORPHOLOGY, ROOT, spatial_blocks

SEEDS = [42, 43, 44]


def load() -> pd.DataFrame:
    b = pd.read_csv(ROOT / "data" / "buildings" / "study_buildings.csv")
    m = pd.read_csv(ROOT / "data" / "validation" / "cadastre" / "study_buildings_cadastre_match.csv")
    if not (np.allclose(b["X_coord"], m["X_coord"]) and np.allclose(b["Y_coord"], m["Y_coord"])):
        raise ValueError("building table and cadastre match are not row-aligned")
    for c in ["iou", "match_dist_m", "cad_current_use", "cad_year_built", "cad_footprint_m2",
              "cad_cadastral_ref"]:
        b[c] = m[c].to_numpy()
    b = b[(b["iou"] > 0.3) & b["cad_current_use"].notna()].copy()
    b["footprint_ok"] = b["match_dist_m"] <= 5.0
    b["cad_residential"] = (b["cad_current_use"] == "1_residential").astype(int)
    b["cad_period"] = pd.cut(b["cad_year_built"], [0, 1940, 1979, 9999], labels=[0, 1, 2]).astype(float)
    b["zoning_RM"] = b["Typology"].isin(["Residential", "Mixed-Use"]).astype(int)
    b["zoning_4class"] = pd.factorize(b["Typology"], sort=True)[0]
    return b


def evaluate(df: pd.DataFrame, target: str, features: list[str], cv: str, seed: int) -> dict:
    d = df.dropna(subset=[target])
    X = d[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = d[target].astype(int).to_numpy()
    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
    if cv == "spatial":
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        proba = cross_val_predict(model, X, y, cv=splitter, groups=spatial_blocks(d, 500.0), method="predict_proba")
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        proba = cross_val_predict(model, X, y, cv=splitter, method="predict_proba")
    pred = proba.argmax(axis=1)
    binary = proba.shape[1] == 2
    return {
        "n": len(y),
        "class_counts": np.bincount(y).tolist(),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "kappa": cohen_kappa_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "roc_auc": roc_auc_score(y, proba[:, 1]) if binary else roc_auc_score(y, proba, multi_class="ovr", average="macro"),
    }


def main():
    df = load()
    rows = []
    samples = {"match_10m": df, "match_10m_footprint_30pct": df[df["footprint_ok"]]}
    feature_sets = {"morphology": MORPHOLOGY, "morphology+location": MORPHOLOGY + LOCATION}
    for sname, sdf in samples.items():
        for target in ["cad_residential", "cad_period", "zoning_RM", "zoning_4class"]:
            for fname, feats in feature_sets.items():
                for cv in ["random", "spatial"]:
                    for seed in SEEDS:
                        rows.append({"sample": sname, "target": target, "features": fname, "cv": cv, "seed": seed,
                                     **evaluate(sdf, target, feats, cv, seed)})
            print(f"{sname} {target}: done")

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "analysis" / "cadastral_target_test.csv", index=False)

    pd.set_option("display.width", 220)
    summary = (res.groupby(["sample", "target", "features", "cv"])
               .agg(n=("n", "first"), kappa=("kappa", "mean"), kappa_sd=("kappa", "std"),
                    bal_acc=("balanced_accuracy", "mean"), auc=("roc_auc", "mean"), f1_macro=("f1_macro", "mean"))
               .round(3))
    print(summary.to_string())
    print("\nClass counts:", res.groupby(["sample", "target"])["class_counts"].first().to_dict())

    for sname, sdf in samples.items():
        k = cohen_kappa_score(sdf["cad_residential"], sdf["zoning_RM"])
        agree = (sdf["cad_residential"] == sdf["zoning_RM"]).mean()
        print(f"\nNo model, {sname}: zoning residential flag vs cadastral residential: agreement {agree:.3f}, kappa {k:.3f}")
        print(pd.crosstab(sdf["zoning_RM"].map({1: "zoning RM", 0: "zoning NR"}),
                          sdf["cad_residential"].map({1: "cadastre residential", 0: "cadastre other"})))


if __name__ == "__main__":
    main()
