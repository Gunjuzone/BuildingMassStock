"""Vectorised Monte Carlo and grouped Sobol engine for floor-area-based material stock models."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
KEY = "Polygon_ID"

MATERIALS = ["Concrete", "Brick", "Wood", "Steel", "Glass", "Plastics", "Aluminium", "Copper"]
CLASSES = ["Residential", "Mixed-Use", "Institutional", "Amenities"]
CLASS_TO_FUNCTION = {"Residential": "RM", "Mixed-Use": "RM", "Institutional": "NR", "Amenities": "NR"}
CLASS_TO_FUNCTION_MU_NR = {**CLASS_TO_FUNCTION, "Mixed-Use": "NR"}
FUNCTIONS = ["RM", "NR"]
STRUCTURES = ["C", "M"]
CATEGORIES = [f"{f}_{s}" for f in FUNCTIONS for s in STRUCTURES]
PERCENTILES = [0, 5, 25, 50, 75, 95, 100]
P50 = PERCENTILES.index(50)

CADASTRAL_STOREY_IQR = (2.095, 3.479)

PERIOD_EDGES = [0, 1950, 1962, 1968, 1974, 1994, 9999]
PERIOD_LABELS = ["<1950", "1950-1962", "1963-1968", "1969-1974", "1975-1994", ">1994"]

GROUP_BLOCKS = {
    "typology": ["typology_zu", "typology_zi"],
    "label": ["label_zu", "label_zi"],
    "structure": ["structure_u", "structure_zg"],
    "height": ["height_zg", "height_zb", "height_zi"],
    "footprint": ["footprint_zg", "footprint_zb", "footprint_zi"],
    "floor_height": ["floor_height_u"],
    "mi": ["mi_zc", "mi_zi"],
}
GROUPS = list(GROUP_BLOCKS)


@dataclass(frozen=True)
class ErrorComponents:
    """Log-scale multiplicative error split into shared and idiosyncratic variance shares."""
    sigma: float
    w_global: float = 0.0
    w_block: float = 0.0

    def __post_init__(self):
        if self.sigma < 0 or self.w_global < 0 or self.w_block < 0 or self.w_global + self.w_block > 1:
            raise ValueError(f"invalid error components: {self}")


@dataclass(frozen=True)
class Config:
    height: ErrorComponents = field(default_factory=lambda: ErrorComponents(0.15))
    footprint: ErrorComponents = field(default_factory=lambda: ErrorComponents(0.15))
    block_size_m: float = 250.0
    mean_unbiased: bool = True
    floor_height_range: tuple[float, float] = (3.0, 3.0)
    round_floors: bool = True

    typology_source: str = "label"
    typology_uncertain: bool = True
    typology_w_unit: float = 0.0
    class_to_function: dict = field(default_factory=lambda: dict(CLASS_TO_FUNCTION))

    label_flip: dict = field(default_factory=lambda: {"RM": 0.0, "NR": 0.0})
    label_w_unit: float = 0.0

    structure_rule: str = "height"
    structure_threshold_m: float = 9.0
    structure_uses_perturbed_height: bool = True
    p_masonry: dict = field(default_factory=lambda: {"RM": 0.30, "NR": 0.30})
    p_masonry_by_period: dict | None = None
    p_masonry_sd: float = 0.0

    mi_mode: str = "stochastic"
    mi_w_category: float = 0.0

    use_recorded_floors: bool = False


def load_buildings(path: Path = ROOT / "data" / "buildings" / "study_buildings.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [KEY, "X_coord", "Y_coord", "Height", "Area", *[f"Prob_{k}" for k in CLASSES]] if c not in df]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if not df[KEY].is_unique:
        raise ValueError(f"{KEY} must be unique")
    return df


def load_rasmi_ranges(path: Path = ROOT / "data" / "rasmi" / "RASMI_EU15.xlsx") -> np.ndarray:
    """RASMI percentiles as an array [category, material, percentile] in kg/m2."""
    q = np.zeros((len(CATEGORIES), len(MATERIALS), len(PERCENTILES)))
    xls = pd.ExcelFile(path)
    for m, material in enumerate(MATERIALS):
        sheet = xls.parse(material)
        for c, category in enumerate(CATEGORIES):
            function, structure = category.split("_")
            row = sheet[(sheet["function"] == function) & (sheet["structure"] == structure)]
            if len(row) != 1:
                raise ValueError(f"{material}: expected one row for {category}, found {len(row)}")
            q[c, m] = row[[f"p_{p}" for p in PERCENTILES]].to_numpy(float)[0]
    if np.any(np.diff(q, axis=2) < 0):
        raise ValueError("RASMI percentiles are not monotone")
    return q


def _factorize(values) -> tuple[np.ndarray, int]:
    codes = pd.factorize(pd.Series(values), sort=True)[0]
    return codes, int(codes.max()) + 1 if len(codes) else 0


class StockModel:
    def __init__(self, buildings: pd.DataFrame, rasmi_q: np.ndarray, config: Config = Config()):
        self.cfg = config
        self.ids = buildings[KEY].to_numpy()
        self.H = buildings["Height"].to_numpy(float)
        self.A = buildings["Area"].to_numpy(float)
        self.n = len(self.H)

        P = buildings[[f"Prob_{c}" for c in CLASSES]].to_numpy(float)
        self.P = P / P.sum(axis=1, keepdims=True)
        self.cumP = np.cumsum(self.P, axis=1)
        self.cumP[:, -1] = 1.0
        if config.typology_source == "label":
            if "Typology" not in buildings:
                raise ValueError("typology_source='label' needs a 'Typology' column")
            self.assigned = buildings["Typology"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
            if np.isnan(self.assigned.astype(float)).any():
                raise ValueError("unknown Typology labels")
        elif config.typology_source == "probabilities":
            self.assigned = self.P.argmax(axis=1)
        else:
            raise ValueError(f"unknown typology_source {config.typology_source!r}")

        xy = buildings[["X_coord", "Y_coord"]].to_numpy(float)
        cells = np.floor((xy - xy.min(axis=0)) / config.block_size_m).astype(int)
        self.block = np.unique(cells, axis=0, return_inverse=True)[1].reshape(-1)
        self.n_blocks = int(self.block.max()) + 1

        unit_col = "Zoning_ID" if "Zoning_ID" in buildings else None
        self.unit, self.n_units = _factorize(buildings[unit_col]) if unit_col else (np.zeros(self.n, int), 1)

        if "period" in buildings:
            self.period = np.array([PERIOD_LABELS.index(p) if p in PERIOD_LABELS else -1
                                    for p in buildings["period"].astype(str)])
        elif "year_built" in buildings:
            self.period = pd.cut(buildings["year_built"], PERIOD_EDGES, labels=False).to_numpy()
            self.period = np.where(np.isnan(self.period.astype(float)), -1, self.period).astype(int)
        else:
            self.period = np.full(self.n, -1)

        self.recorded_floors = (buildings["recorded_floors"].to_numpy(float)
                                if "recorded_floors" in buildings else None)
        if config.use_recorded_floors and self.recorded_floors is None:
            raise ValueError("use_recorded_floors requires a 'recorded_floors' column")

        self.q = rasmi_q
        self.class_function = np.array([FUNCTIONS.index(config.class_to_function[c]) for c in CLASSES])
        self.p_masonry = self._masonry_probabilities()
        self.label_flip = np.array([config.label_flip.get(f, 0.0) for f in FUNCTIONS])

    def _masonry_probabilities(self) -> np.ndarray:
        """Masonry probability per building and function: shape (n, 2)."""
        cfg = self.cfg
        base = np.array([cfg.p_masonry[f] for f in FUNCTIONS])
        p = np.tile(base, (self.n, 1))
        if cfg.p_masonry_by_period:
            for label, value in cfg.p_masonry_by_period.items():
                if label not in PERIOD_LABELS:
                    raise ValueError(f"unknown period label {label!r}")
                p[self.period == PERIOD_LABELS.index(label)] = value
        return p

    def draw(self, n_iter: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        n, nb, nu = self.n, self.n_blocks, self.n_units
        return {
            "typology_zu": rng.standard_normal((n_iter, nu)),
            "typology_zi": rng.standard_normal((n_iter, n)),
            "label_zu": rng.standard_normal((n_iter, nu)),
            "label_zi": rng.standard_normal((n_iter, n)),
            "structure_u": rng.random((n_iter, n)),
            "structure_zg": rng.standard_normal((n_iter, 1)),
            "height_zg": rng.standard_normal((n_iter, 1)),
            "height_zb": rng.standard_normal((n_iter, nb)),
            "height_zi": rng.standard_normal((n_iter, n)),
            "footprint_zg": rng.standard_normal((n_iter, 1)),
            "footprint_zb": rng.standard_normal((n_iter, nb)),
            "footprint_zi": rng.standard_normal((n_iter, n)),
            "floor_height_u": rng.random((n_iter, 1)),
            "mi_zc": rng.standard_normal((n_iter, len(CATEGORIES))),
            "mi_zi": rng.standard_normal((n_iter, n)),
        }

    def _multiplier(self, comp: ErrorComponents, zg, zb, zi) -> np.ndarray:
        w_i = 1.0 - comp.w_global - comp.w_block
        e = comp.sigma * (np.sqrt(comp.w_global) * zg + np.sqrt(comp.w_block) * zb[:, self.block] + np.sqrt(w_i) * zi)
        if self.cfg.mean_unbiased:
            e = e - comp.sigma ** 2 / 2
        return np.exp(e)

    def _unit_uniform(self, zu, zi, w_unit: float) -> np.ndarray:
        """Uniform draw per building, sharing a fraction w_unit of its variance within a zoning unit."""
        z = np.sqrt(w_unit) * zu[:, self.unit] + np.sqrt(1.0 - w_unit) * zi
        return ndtr(z)

    def _mi_quantile(self, category: np.ndarray, u: np.ndarray) -> np.ndarray:
        out = np.empty(category.shape + (len(MATERIALS),))
        xp = np.array(PERCENTILES) / 100.0
        for c in range(len(CATEGORIES)):
            mask = category == c
            if not mask.any():
                continue
            uc = u[mask]
            for m in range(len(MATERIALS)):
                out[..., m][mask] = np.interp(uc, xp, self.q[c, m])
        return out

    def _structure_is_masonry(self, b, function, height) -> np.ndarray:
        cfg = self.cfg
        p = self.p_masonry[np.arange(self.n)[None, :], function]
        if cfg.p_masonry_sd:
            p = np.clip(p + cfg.p_masonry_sd * b["structure_zg"], 0.0, 1.0)
        masonry = b["structure_u"] < p
        if cfg.structure_rule in ("height", "period_height"):
            masonry &= height <= cfg.structure_threshold_m
        elif cfg.structure_rule != "period":
            raise ValueError(f"unknown structure_rule {cfg.structure_rule!r}")
        return masonry

    def masses(self, b: dict[str, np.ndarray]) -> np.ndarray:
        """Material mass (kg) per iteration, building and material."""
        cfg = self.cfg
        shape = b["typology_zi"].shape

        if cfg.typology_uncertain and cfg.typology_source == "probabilities":
            u = self._unit_uniform(b["typology_zu"], b["typology_zi"], cfg.typology_w_unit)
            cls = np.minimum((u[..., None] > self.cumP[None]).sum(axis=-1), len(CLASSES) - 1)
        else:
            cls = np.broadcast_to(self.assigned, shape)
        function = self.class_function[cls]

        if self.label_flip.any():
            u_label = self._unit_uniform(b["label_zu"], b["label_zi"], cfg.label_w_unit)
            function = np.where(u_label < self.label_flip[function], 1 - function, function)

        H = self.H * self._multiplier(cfg.height, b["height_zg"], b["height_zb"], b["height_zi"])
        A = self.A * self._multiplier(cfg.footprint, b["footprint_zg"], b["footprint_zb"], b["footprint_zi"])
        if cfg.use_recorded_floors:
            floors = np.broadcast_to(np.maximum(1.0, self.recorded_floors), shape)
        else:
            lo, hi = cfg.floor_height_range
            floors = H / (lo + (hi - lo) * b["floor_height_u"])
            floors = np.maximum(1.0, np.round(floors) if cfg.round_floors else floors)
        gfa = A * floors

        h_struct = H if cfg.structure_uses_perturbed_height else np.broadcast_to(self.H, shape)
        masonry = self._structure_is_masonry(b, function, h_struct)
        category = function * len(STRUCTURES) + masonry.astype(int)

        if cfg.mi_mode == "p50":
            mi = self.q[:, :, P50][category]
        elif cfg.mi_mode == "stochastic":
            z = (np.sqrt(cfg.mi_w_category) * np.take_along_axis(b["mi_zc"], category, axis=1)
                 + np.sqrt(1.0 - cfg.mi_w_category) * b["mi_zi"])
            mi = self._mi_quantile(category, ndtr(z))
        else:
            raise ValueError(f"unknown mi_mode {cfg.mi_mode!r}")
        return gfa[..., None] * mi

    def evaluate(self, blocks: dict[str, np.ndarray], keep_buildings: bool = False, chunk: int = 250):
        n_iter = blocks["typology_zi"].shape[0]
        totals = np.zeros((n_iter, len(MATERIALS)))
        per_building = np.zeros((n_iter, self.n, len(MATERIALS)), dtype=np.float32) if keep_buildings else None
        for start in range(0, n_iter, chunk):
            sl = slice(start, start + chunk)
            mass = self.masses({k: v[sl] for k, v in blocks.items()})
            totals[sl] = mass.sum(axis=1)
            if keep_buildings:
                per_building[sl] = mass
        return totals, per_building

    def deterministic(self, floor_height: float = 3.0) -> np.ndarray:
        """Single-label baseline: most probable class, observed geometry, p50 MI."""
        function = self.class_function[self.assigned]
        if self.cfg.use_recorded_floors:
            floors = np.maximum(1.0, self.recorded_floors)
        else:
            floors = np.maximum(1.0, np.round(self.H / floor_height))
        gfa = self.A * floors
        q50 = self.q[:, :, P50]
        p_m = self.p_masonry[np.arange(self.n), function]
        if self.cfg.structure_rule in ("height", "period_height"):
            p_m = np.where(self.H <= self.cfg.structure_threshold_m, p_m, 0.0)
        base = function * len(STRUCTURES)
        mi = (1 - p_m)[:, None] * q50[base] + p_m[:, None] * q50[base + 1]
        return gfa[:, None] * mi


def run_monte_carlo(model: StockModel, n_iter: int, seed: int, keep_buildings: bool = True):
    blocks = model.draw(n_iter, np.random.default_rng(seed))
    return model.evaluate(blocks, keep_buildings=keep_buildings)


def summarise(totals: np.ndarray, per_building: np.ndarray | None, deterministic: np.ndarray) -> pd.DataFrame:
    mean = totals.mean(axis=0)
    sd = totals.std(axis=0, ddof=1)
    out = pd.DataFrame({
        "material": MATERIALS,
        "district_mean_Mkg": mean / 1e6,
        "district_p5_Mkg": np.percentile(totals, 5, axis=0) / 1e6,
        "district_p95_Mkg": np.percentile(totals, 95, axis=0) / 1e6,
        "district_cv_pct": 100 * sd / mean,
        "deterministic_Mkg": deterministic.sum(axis=0) / 1e6,
    })
    if per_building is not None:
        b_mean = per_building.mean(axis=0, dtype=np.float64)
        b_cv = 100 * per_building.std(axis=0, ddof=1, dtype=np.float64) / np.where(b_mean > 0, b_mean, np.nan)
        out["building_cv_mean_pct"] = np.nanmean(b_cv, axis=0)
        out["building_cv_median_pct"] = np.nanmedian(b_cv, axis=0)
    return out


def cv_by_sample_size(per_building: np.ndarray, sizes, n_rep: int, rng: np.random.Generator) -> pd.DataFrame:
    """CV of the summed stock for random building subsets, reusing one Monte Carlo run."""
    rows = []
    n = per_building.shape[1]
    for size in sizes:
        for rep in range(n_rep):
            idx = rng.choice(n, size=size, replace=False)
            tot = per_building[:, idx, :].sum(axis=1, dtype=np.float64)
            cv = 100 * tot.std(axis=0, ddof=1) / tot.mean(axis=0)
            rows += [{"sample_size": size, "replicate": rep, "material": m, "cv_pct": cv[k]} for k, m in enumerate(MATERIALS)]
    return pd.DataFrame(rows)


def _indices(ya, yb, yab):
    """First-order (Saltelli 2010) and total-order (Jansen 1999) indices, mean-centred."""
    mu = np.mean(np.concatenate([ya, yb], axis=0), axis=0)
    ya, yb, yab = ya - mu, yb - mu, yab - mu
    var = np.var(np.concatenate([ya, yb], axis=0), axis=0, ddof=1)
    var = np.where(var > 0, var, np.nan)
    return np.mean(yb * (yab - ya), axis=0) / var, 0.5 * np.mean((ya - yab) ** 2, axis=0) / var


def sobol_grouped(model: StockModel, n_base: int, seed: int, groups=GROUPS, n_boot: int = 200,
                  building_level: bool = False, second_order: bool = False):
    """Grouped Sobol indices, with bootstrap intervals and optional pairwise (second-order) terms."""
    rng = np.random.default_rng(seed)
    A = model.draw(n_base, rng)
    B = model.draw(n_base, rng)
    yA, bA = model.evaluate(A, keep_buildings=building_level)
    yB, bB = model.evaluate(B, keep_buildings=building_level)
    boot = rng.integers(0, n_base, size=(n_boot, n_base))

    def freeze(names):
        keys = {k for g in names for k in GROUP_BLOCKS[g]}
        return {k: (B[k] if k in keys else A[k]) for k in A}

    district_rows, building_rows, first_order = [], [], {}
    for g in groups:
        yAB, bAB = model.evaluate(freeze([g]), keep_buildings=building_level)
        s1, st = _indices(yA, yB, yAB)
        first_order[g] = s1
        bs1, bst = zip(*(_indices(yA[i], yB[i], yAB[i]) for i in boot))
        lo1, hi1 = np.percentile(bs1, [2.5, 97.5], axis=0)
        lot, hit = np.percentile(bst, [2.5, 97.5], axis=0)
        for k, m in enumerate(MATERIALS):
            district_rows.append({"group": g, "material": m, "S1": s1[k], "S1_lo": lo1[k], "S1_hi": hi1[k],
                                  "ST": st[k], "ST_lo": lot[k], "ST_hi": hit[k]})
        if building_level:
            b_s1, b_st = _indices(bA.astype(np.float64), bB.astype(np.float64), bAB.astype(np.float64))
            for k, m in enumerate(MATERIALS):
                building_rows.append({"group": g, "material": m,
                                      "S1_median": np.nanmedian(b_s1[:, k]),
                                      "S1_p25": np.nanpercentile(b_s1[:, k], 25), "S1_p75": np.nanpercentile(b_s1[:, k], 75),
                                      "ST_median": np.nanmedian(b_st[:, k]),
                                      "ST_p25": np.nanpercentile(b_st[:, k], 25), "ST_p75": np.nanpercentile(b_st[:, k], 75)})

    pair_rows = []
    if second_order:
        for g1, g2 in combinations(groups, 2):
            yAB, _ = model.evaluate(freeze([g1, g2]))
            closed, _ = _indices(yA, yB, yAB)
            s2 = closed - first_order[g1] - first_order[g2]
            for k, m in enumerate(MATERIALS):
                pair_rows.append({"group_1": g1, "group_2": g2, "material": m,
                                  "S_closed": closed[k], "S2_interaction": s2[k]})

    district = pd.DataFrame(district_rows)
    district["interaction_ST_minus_S1"] = district["ST"] - district["S1"]
    return (district,
            pd.DataFrame(building_rows) if building_level else None,
            pd.DataFrame(pair_rows) if second_order else None)


if __name__ == "__main__":
    buildings = load_buildings()
    q = load_rasmi_ranges()
    model = StockModel(buildings, q, Config())
    totals, per_b = run_monte_carlo(model, n_iter=500, seed=42)
    pd.set_option("display.width", 200)
    print(summarise(totals, per_b, model.deterministic()).round(2).to_string(index=False))
