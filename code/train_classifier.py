"""Train the functional-typology classifier from the zoning designations."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Residential", "Mixed-Use", "Institutional", "Amenities"]
MORPHOLOGY = ["Area", "Perimeter", "MBG_Width", "MBG_Length", "Aspect_Ratio", "Height",
              "Convexity", "Solidity", "Elongation", "Rectangularity", "Form_ratio"]
LOCATION = ["X_coord", "Y_coord", "GLH"]
EXCLUDED_LEAKY = ["Match_Score", "Confidence_Score"]
KEY = "Polygon_ID"


def spatial_blocks(df: pd.DataFrame, size_m: float) -> np.ndarray:
    xy = df[["X_coord", "Y_coord"]].to_numpy(float)
    cells = np.floor((xy - xy.min(axis=0)) / size_m).astype(int)
    return np.unique(cells, axis=0, return_inverse=True)[1].reshape(-1)


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    logp = np.log(np.clip(probs, 1e-12, 1.0))

    def nll(t):
        return -np.mean(log_softmax(logp / t, axis=1)[np.arange(len(y)), y])

    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    return np.exp(log_softmax(np.log(np.clip(probs, 1e-12, 1.0)) / t, axis=1))


def reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                         "mean_confidence": conf[m].mean(), "accuracy": correct[m].mean()})
    return float(ece), pd.DataFrame(rows)


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.sum((probs - np.eye(len(CLASSES))[y]) ** 2, axis=1)))


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))))


def fit_predict(backend: str, X_fit, y_fit, X_eval: list, args, workdir: Path) -> list[np.ndarray]:
    """Train on (X_fit, y_fit) and return class probabilities (CLASSES order) for each X_eval."""
    if backend == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(random_state=args.seed).fit(X_fit, y_fit)
        order = [list(model.classes_).index(k) for k in range(len(CLASSES))]
        return [model.predict_proba(X)[:, order] for X in X_eval]

    from autogluon.tabular import TabularPredictor
    cols = [f"f{j}" for j in range(X_fit.shape[1])]
    train = pd.DataFrame(X_fit, columns=cols)
    train["label"] = np.array(CLASSES)[y_fit]
    try:
        predictor = TabularPredictor(label="label", path=str(workdir), eval_metric="log_loss", verbosity=0)
        predictor.fit(train, presets=args.preset, time_limit=args.time_limit, calibrate=False)
        return [predictor.predict_proba(pd.DataFrame(X, columns=cols))[CLASSES].to_numpy() for X in X_eval]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=ROOT / "data" / "buildings" / "study_buildings.csv")
    ap.add_argument("--backend", choices=["autogluon", "hgb"], default="autogluon")
    ap.add_argument("--features", choices=["morphology", "morphology+location"], default="morphology")
    ap.add_argument("--cv", choices=["zoning", "spatial", "random"], default="zoning")
    ap.add_argument("--block-size", type=float, default=500.0, help="spatial CV block size (m)")
    ap.add_argument("--preset", default="best_quality")
    ap.add_argument("--time-limit", type=int, default=1440, help="AutoGluon seconds per outer fold")
    ap.add_argument("--no-smote", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if not df[KEY].is_unique:
        raise ValueError(f"{KEY} must be unique")
    features = MORPHOLOGY + (LOCATION if args.features == "morphology+location" else [])
    assert not set(features) & set(EXCLUDED_LEAKY)
    X_all = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = df["Typology"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    if np.isnan(y.astype(float)).any():
        raise ValueError("unknown Typology labels")
    if args.cv == "zoning":
        groups = pd.factorize(df["Zoning_ID"])[0]
    else:
        groups = spatial_blocks(df, args.block_size)

    out_dir = ROOT / "data" / "classifier" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cv in ("zoning", "spatial"):
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed).split(X_all, y, groups)
    else:
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(X_all, y)

    n = len(df)
    p_cal_oof = np.full((n, len(CLASSES)), np.nan)
    p_raw_oof = np.full((n, len(CLASSES)), np.nan)
    fold_of = np.full(n, -1)
    fold_rows = []

    for k, (tr, te) in enumerate(outer):
        if args.cv in ("zoning", "spatial"):
            inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed + k).split(X_all[tr], y[tr], groups[tr])
        else:
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed + k).split(X_all[tr], y[tr])
        i_fit, i_cal = next(inner)
        fit, cal = tr[i_fit], tr[i_cal]

        medians = np.nanmedian(X_all[fit], axis=0)
        X = np.where(np.isnan(X_all), medians, X_all)
        scaler = StandardScaler().fit(X[fit])
        X_fit, X_cal, X_te = scaler.transform(X[fit]), scaler.transform(X[cal]), scaler.transform(X[te])
        y_fit = y[fit]
        if not args.no_smote:
            from imblearn.over_sampling import SMOTE
            k_nn = max(1, min(5, np.bincount(y_fit).min() - 1))
            X_fit, y_fit = SMOTE(random_state=args.seed, k_neighbors=k_nn).fit_resample(X_fit, y_fit)

        workdir = Path(tempfile.mkdtemp(prefix=f"ag_fold{k}_")) if args.backend == "autogluon" else Path()
        p_cal_raw, p_te_raw = fit_predict(args.backend, X_fit, y_fit, [X_cal, X_te], args, workdir)
        t = fit_temperature(p_cal_raw, y[cal])
        p_te = apply_temperature(p_te_raw, t)

        p_raw_oof[te], p_cal_oof[te], fold_of[te] = p_te_raw, p_te, k
        pred = p_te.argmax(axis=1)
        ece_raw, _ = reliability(p_te_raw, y[te])
        ece_cal, _ = reliability(p_te, y[te])
        fold_rows.append({"fold": k, "n_fit": len(fit), "n_cal": len(cal), "n_test": len(te),
                          "accuracy": accuracy_score(y[te], pred), "f1_macro": f1_score(y[te], pred, average="macro"),
                          "kappa": cohen_kappa_score(y[te], pred), "temperature": t,
                          "ece_uncalibrated": ece_raw, "ece_calibrated": ece_cal})
        print(fold_rows[-1])

    assert (fold_of >= 0).all()
    pred = p_cal_oof.argmax(axis=1)
    ece_raw, rel_raw = reliability(p_raw_oof, y)
    ece_cal, rel_cal = reliability(p_cal_oof, y)
    summary = {
        "run_name": args.run_name, "backend": args.backend, "preset": args.preset, "features": features,
        "cv": args.cv, "block_size_m": args.block_size, "smote": not args.no_smote, "seed": args.seed,
        "accuracy": accuracy_score(y, pred), "f1_macro": f1_score(y, pred, average="macro"),
        "f1_weighted": f1_score(y, pred, average="weighted"), "kappa": cohen_kappa_score(y, pred),
        "f1_per_class": dict(zip(CLASSES, f1_score(y, pred, average=None, labels=range(len(CLASSES))).round(4).tolist())),
        "ece_uncalibrated": ece_raw, "ece_calibrated": ece_cal,
        "brier_uncalibrated": brier(p_raw_oof, y), "brier_calibrated": brier(p_cal_oof, y),
        "log_loss_uncalibrated": nll(p_raw_oof, y), "log_loss_calibrated": nll(p_cal_oof, y),
        "temperatures": [r["temperature"] for r in fold_rows],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.concat([rel_raw.assign(probabilities="uncalibrated"), rel_cal.assign(probabilities="calibrated")]).to_csv(
        out_dir / "reliability.csv", index=False)
    pd.DataFrame(confusion_matrix(y, pred, labels=range(len(CLASSES))), index=CLASSES, columns=CLASSES).to_csv(
        out_dir / "confusion_matrix.csv")

    oof = pd.DataFrame({KEY: df[KEY], "fold": fold_of, "Typology": df["Typology"], "Predicted": np.array(CLASSES)[pred]})
    for j, c in enumerate(CLASSES):
        oof[f"Prob_{c}"] = p_cal_oof[:, j]
        oof[f"ProbRaw_{c}"] = p_raw_oof[:, j]
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)

    engine_cols = [KEY, "X_coord", "Y_coord", "Height", "Area", "Typology"]
    buildings = df[engine_cols].merge(oof[[KEY, "Predicted"] + [f"Prob_{c}" for c in CLASSES]], on=KEY, how="left", validate="one_to_one")
    assert buildings[[f"Prob_{c}" for c in CLASSES]].notna().all().all()
    buildings.to_csv(ROOT / "data" / "buildings" / f"buildings_{args.run_name}.csv", index=False)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("features",)}, indent=2))


if __name__ == "__main__":
    main()
