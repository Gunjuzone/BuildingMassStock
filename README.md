# Propagation of typology misclassification and geometric measurement errors in floor-area-based urban material stock models

Code for a Monte Carlo framework that propagates three sources of uncertainty through a floor-area
material-intensity model of urban building stocks: the error in the functional designation taken from
zoning records, the error in remotely sensed building height and footprint area, and the spread of the
material intensity coefficients themselves. The study area is 1,746 buildings across three districts of
Barcelona, covered by a Maxar Precision3D acquisition.

What distinguishes the treatment of error here is that its magnitude and its correlation between
neighbouring buildings are measured against independent cadastral records for the same buildings, instead
of assumed. Over 872 matched buildings the modelled floor area departs from the record with a log-scale
standard deviation of 0.50, more than three times the 0.15 a study takes from the remote sensing
literature, and the residuals are correlated within about 100 m.

## The model is defined once

`code/baseline.py` holds the single definition of every configuration used in the paper, and every
experiment imports from it, so no script can build a configuration of its own.

| Function | What it is |
|---|---|
| `submitted()` | the simpler design kept for comparison: 9 m height threshold, flat masonry share, no label error, assumed sigma 0.15 |
| `conditional()` | the reference model: structure drawn by construction period, measured label error, fitted error magnitude and correlation, MI at its median |
| `baseline()` | `conditional()` with material intensity sampled and half shared within a category |
| `classifier_probs()` | typology sampled from the calibrated classifier probabilities instead of the zoning label |

## Running it

```
pip install -r requirements.txt

python code/parse_cadastre.py            # INSPIRE Buildings GML for Barcelona
python code/match_by_overlap.py          # pair study buildings with cadastral records
python code/build_buildings_table.py     # the table the engine reads
python code/fit_error_model.py           # measure the error magnitude and its spatial correlation
python code/train_classifier.py --backend autogluon --cv zoning --features morphology --run-name ag_zoning_morphology

python code/run_reanalysis.py --tag baseline_v3      # E1 to E11
python code/class_breakdown.py --tag baseline_v3     # E12
TAG=baseline_v3 python code/run_sobol_classifier_probs.py
python code/convergence_check.py --tag baseline_v3   # E14
python code/error_model_robustness.py --tag baseline_v3  # E15
python code/masonry_share_sweep.py --tag baseline_v3 # E16
```

The figures in the paper are not drawn here. Every value behind them is in SI2, sheet by sheet, and in
`analysis/results/<tag>/`.

## Data

The Maxar Precision3D building vectors are licensed and cannot be redistributed. The cadastral records come
from the INSPIRE Buildings service of the Dirección General del Catastro, which is the author and owner of
that data. Zoning records are from Ajuntament de Barcelona. Material intensity coefficients are from RASMI
(Fishman et al., 2024), EU15 region.

Scripts that read the licensed inputs will not run without them. Everything downstream of
`data/buildings/study_buildings_enriched.csv` is reproducible from the results tables.

The MIT licence in `LICENSE` covers the code in this repository. It does not extend to the input data,
which are licensed by the bodies named above.

`code/make_synthetic_table.py` writes a table of the same shape with values drawn from simple
distributions, together with the label-error rates that go with it, so the simulation and the
decomposition can be run without the licensed inputs:

```
python code/make_synthetic_table.py
python code/run_reanalysis.py --buildings data/buildings/study_buildings_synthetic.csv --tag synthetic
BUILDINGS=data/buildings/study_buildings_synthetic.csv python code/class_breakdown.py --tag synthetic
```

It reproduces no number in the paper. It exists so the code can be run end to end.

## Other scripts

| Script | What it does |
|---|---|
| `code/stock_engine.py` | the Monte Carlo and grouped Sobol engine every experiment calls |
| `code/error_model_robustness.py` | the measured error under other pairing rules |
| `code/cadastre_study_area.py` | summary of the cadastre clipped to the study districts |
| `code/test_cadastral_targets.py` | whether morphology predicts cadastral use or construction period |

## Method notes

- Geometric error is a mean-one lognormal multiplier on height and footprint area, split into a component
  common to every building, a component common to a 500 m grid cell, and a per-building component, with the
  three shares estimated from the cadastral residuals.
- Structure is drawn against the masonry share for each building's construction period, taken from Lantada
  (2007); the 957 buildings with no recorded year take the citywide share of 0.79.
- Material intensity is drawn by reading the reported RASMI percentiles at a random probability, with a
  parameter governing how far that draw is shared within a category.
- Sobol indices are grouped, first-order after Saltelli et al. (2010) and total-order after Jansen (1999),
  with bootstrap confidence intervals.
