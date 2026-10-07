# Pairing AIS with RF:<br>Analysis of a Markovian Approach for Maritime Data Fusion

## Objective

Maritime transport carries around 90% of global trade but faces congestion, safety risks, and illicit activities. Authorities rely on the Automatic Identification System (AIS) for vessel tracking, yet AIS suffers from coverage gaps and manipulation vulnerabilities. Space-based radio frequency (RF) detections offer a complementary source, but methods to fuse AIS and RF are underdeveloped. This project matches single synthetic RF detections to AIS trajectories with a Pair Hidden Markov Model (PHMM)-style model, compares it with nearest-neighbor baselines in top-1 precision and in detecting dark vessels, and analyzes its errors with bias tests. 

## Reproducing results

All published results use the **gaussian** RF error model. Setup (uv, Python 3.11) is described in [Setup and data](#setup-and-data).

```bash
# 1) Environment
uv sync --frozen

# 2) Base data (once): download AIS, filter cargo vessels, run
#    notebooks/02_data_preparation.ipynb  (details below)

# 3) Everything the paper reports, gaussian model (long, the Forward scoring dominates)
uv run python -m scripts.run_pipeline

# 4) Redraw figures only, after changing plotting code
uv run python -m scripts.run_pipeline --plots-only
```

`run_pipeline` runs preselection, the PHMM Forward scores (including the extra pass with kappa=(1,1) for the ablation), the four nearest-neighbor baselines, and all evaluation scripts. Results land in `reports/tables/<phase>/gaussian/` and `reports/figures/<phase>/gaussian/` (gitignored, regenerated on demand). 

### Where each paper result comes from

Each evaluation script prints a one-line summary and writes a small set of result CSVs to `reports/tables/<phase>/gaussian/`. Point-level figure inputs are stored separately in `reports/plot_data/` (one pickle per phase) and are not results.

| Paper element | Source (`reports/tables/...`) |
|---|---|
| Table II, RF detections (17,081 full, 10,903 single-candidate, 6,178 multi-candidate, Forward alpha=1 precision) | `phase4_evaluation/stratified_ranking.csv` (`recall@1`) |
| Table II, tuning split (1,236, precision 0.7921) and validation split (4,942) | `phase4_evaluation/alpha_search.csv`, `phase4b_baseline_comparison/ranking_metrics.csv` (`n_points`) |
| Table II, trajectories (3,203, 1,022 MMSIs, known/dark split, 3,330 dark detections) | `phase4c_open_set_evaluation/diagnostics.csv` |
| Recall@3/@5 | `phase4b_baseline_comparison/ranking_metrics.csv` (population `multimatch`) |
| 203 tied signals (raw log-odds, validation split) | `phase4_evaluation/diagnostics.csv` |
| Table III, bias tests 1-4 | `phase4c_bias_experiments/bias_experiments_summary.csv` (log-odds column is the n^alpha variant) |
| Table IV, precision of all rows except Forward (alpha=1), random baseline | `phase4b_baseline_comparison/metrics.csv` (population `multimatch`) |
| Table IV, Forward (alpha=1) and the kappa ablation | `phase4_evaluation/kappa_sensitivity.csv` |
| Table IV, AUC (dark-vessel test) and the 70.45% prefilter rejection rate | `phase4c_open_set_evaluation/open_set_summary_darkvessel.csv` (`auc_scored_only_raw`, `automatic_rejection_rate`) |
| McNemar tests | `phase4b_baseline_comparison/mcnemar_test.csv` |
| Tuned alpha (Forward 0.90, log-odds 1.27) | `phase4_evaluation/alpha_search.csv` |

All other CSVs are supporting detail: `ttest_results.csv` and `confusion_matrices.csv` (phase 4), `bootstrap_ci.csv` (phase 4b) and a `diagnostics.csv` per phase with counts and diagnostics that were previously only printed.

### Figures used in the paper

Figures are in `reports/figures/<phase>/gaussian/`.

| Paper figure | File |
|---|---|
| Fig. 4a, trajectory length, PHMM Forward (corrected) | `phase4c_bias_experiments/experiment_1_points_boxplot_phmm_forward_corrected.png` |
| Fig. 4b, trajectory length, PHMM log-odds (n^alpha) | `phase4c_bias_experiments/experiment_1_points_boxplot_log_odds_phmm_n_alpha.png` |
| Fig. 4c, trajectory length, NN time-weighted | `phase4c_bias_experiments/experiment_1_points_boxplot_nn_time_weighted.png` |
| Fig. 5, ROC of the dark-vessel test | `phase4c_open_set_evaluation/roc_scored_only_darkvessel.png` |

All other PNGs in `reports/figures/` are additional diagnostics.

## Setup and data
This project uses **uv** for environment + dependency management (https://pypi.org/project/uv/)
- **Python version**: see `.python-version` in the repo (3.11)

### Step 1 — Install `uv`

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Afterward, you can verify the installation by running uv version:
$ uv version
```

### Step 2 — Clone this repository

```bash
git clone https://github.com/Cayah99/AIS_RF_matching.git
cd AIS_RF_matching
```

### Step 3 — Create & sync the environment

```bash
uv sync --frozen
```

### Step 4 — (optional) Activate the virtualenv

```bash
# On macOS and Linux.
source .venv/bin/activate

# On Windows.
.\.venv\Scripts\activate.bat
```

### Step 5 — Reproduce the datasets

```bash
# Make data folder 

# On macOS and Linux.
mkdir data/raw

# On Windows.
mkdir data\raw

# 1) Download AIS data  (~15 minutes, depending on connection and CPU)
# Source: NOAA’s Marine Cadastre (https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/index.html)
# Run:
    uv run python scripts/data_loading/download_AIS_data.py
# Output: data/raw/AIS_01_2024.pkl

# 2) Filter op cargotypes
# Run:  
    uv run python scripts/data_preprocessing/AIS/pre_filter_vessel_type.py
# Output: data/processed/cargo_vessels.parquet

# 3) Preprocess data, including synthetic RF signal generation
# Run full '02_data_preparation.ipynb' notebook
    uv run python -m notebook notebooks/02_data_preparation.ipynb
# Output: 
    # 1. data/processed/AIS_sample_no_RF_5000.pkl
    # 2. data/processed/statistics_sample_5000.pkl
    # 3. data/processed/uniform/train_data_sample_5000.pkl        (RF error model: uniform)
    # 4. data/processed/gaussian/train_data_sample_5000.pkl       (RF error model: gaussian)

# 4) Preselect AIS–RF candidates 
# Fast (seconds on the sample dataset, parallelized across all CPU cores).
# Defaults to --error-model gaussian (the model used in the paper).
    uv run python scripts/modeling/AIS_RF_preselection/AIS_RF_preselection.py
# Output: data/processed/gaussian/AIS_RF_preselection_data.pkl

# 5) Forward alignment scores (PHMM) 
# NOTE: On the full study dataset, this step is compute-intensive (the longest step of the pipeline).
# Progress is checkpointed per AIS track, so an interrupted run can be resumed
# by simply re-running the same command.
# Run (defaults to --error-model gaussian):
    uv run python scripts/modeling/AIS_RF_alignment/AIS_RF_forward_alignment.py
# Output: data/processed/gaussian/AIS_RF_forward_scores_data.pkl
# Checkpoint: data/processed/gaussian/AIS_RF_forward_scores_checkpoint.pkl

# 6) Nearest-neighbor baselines
# Runs four lower-bound baselines to compare the PHMM Forward alignment
# against: plain Euclidean, Haversine (great-circle), point-to-segment
# (perpendicular distance to the nearest AIS track leg), and time-weighted
# (point-to-segment distance combined with an interpolated time gap, using
# the same distance_threshold=6km/time_window_hours=3 defaults as the
# preselection step). Fast (seconds to ~2 minutes on the sample dataset);
# progress is checkpointed per AIS track like the Forward alignment step above.
# Run (defaults to --error-model gaussian):
    uv run python scripts/modeling/AIS_RF_alignment/AIS_RF_nn_baseline.py
# Output:
    # data/processed/gaussian/AIS_RF_nn_baseline_euclidean_scores_data.pkl
    # data/processed/gaussian/AIS_RF_nn_baseline_haversine_scores_data.pkl
    # data/processed/gaussian/AIS_RF_nn_baseline_segment_scores_data.pkl
    # data/processed/gaussian/AIS_RF_nn_baseline_time_weighted_scores_data.pkl
```

### Full pipeline options

`scripts/run_pipeline.py` runs preselection through baseline scoring and
every phase 4 report needed for the paper (top-1 evaluation, baseline
comparison, dark-vessel detection, misclassification-bias
significance experiments, kappa sensitivity ablation), cleaning previously
generated outputs first. It does **not** regenerate the base sample data --
run `02_data_preparation.ipynb` first ("Reproduce the datasets" above).

```bash
# gaussian error model (default), everything except leave-one-out
uv run python -m scripts.run_pipeline

# Both error models
uv run python -m scripts.run_pipeline --error-model both

# One error model only
uv run python -m scripts.run_pipeline --error-model uniform

# Also run the leave-one-out open-set negatives (a second, independent
# recompute of preselection+scoring with each RF point's own track
# excluded; off by default -- the primary dark-vessel/MMSI open-set check
# always runs regardless)
uv run python -m scripts.run_pipeline --leave-one-out
```

### RF error models

`02_data_preparation.ipynb` simulates RF direction-finding detections around
each vessel's true AIS position using one of two error models, both defined in
`scripts/data_preprocessing/RF/generate_synthethic_RF_data.py`:

- **`uniform`** (default): the RF bearing is the vessel heading plus uniform
  noise in `[-error_bearing, +error_bearing]` (default ±60°), and the offset
  distance is drawn from `Normal(mean_distance=2000, std_dev_distance=100)`
  meters — i.e. a fixed ~2 km systematic offset in a heading-relative cone.
- **`gaussian`**: an isotropic 2D positional error with no heading dependence
  and no fixed offset. East/north components are each drawn from
  `Normal(0, sigma)` (default `sigma=800` meters), so the resulting distance
  follows a Rayleigh distribution (mean ≈ 1.25 × sigma) and the bearing is
  uniform over the full circle.

The notebook generates both variants in the same run, each into its own
subfolder: `uniform` output goes to `data/processed/uniform/`, `gaussian`
output to `data/processed/gaussian/`. Downstream scripts select which one to
use via `--error-model` (default: `gaussian`):

```bash
# Uses data/processed/uniform/
uv run python scripts/modeling/AIS_RF_preselection/AIS_RF_preselection.py --error-model uniform

# Uses data/processed/gaussian/ (default)
uv run python scripts/modeling/AIS_RF_preselection/AIS_RF_preselection.py --error-model gaussian
uv run python scripts/modeling/AIS_RF_alignment/AIS_RF_forward_alignment.py --error-model gaussian
uv run python scripts/modeling/AIS_RF_alignment/AIS_RF_nn_baseline.py --error-model gaussian
```

### Phase 4 evaluation experiments

`scripts/evaluation/run_phase4*.py` are repeatable, CLI-driven versions of the
Phase 4 evaluation notebooks: each one loads the data for one `--error-model`,
runs its full battery of metrics/experiments/significance tests, logs a
one-line summary, and — instead of `plt.show()` — saves every figure to
`reports/figures/<script>/<error_model>/`, the few result tables to
`reports/tables/<script>/<error_model>/` and the figure inputs to
`reports/plot_data/<script>/<error_model>.pkl`. `reports/` is gitignored, like the
`data/processed/*.pkl` pipeline outputs — it is regenerated on demand, not
committed.

```bash
uv run python -m scripts.evaluation.run_phase4_evaluation --error-model uniform
uv run python -m scripts.evaluation.run_phase4b_baseline_comparison --error-model uniform
uv run python -m scripts.evaluation.run_phase4c_open_set_evaluation --error-model gaussian
uv run python -m scripts.evaluation.run_phase4c_bias_experiments --error-model uniform
```

The corresponding notebooks (`04_evaluation.ipynb` through
`07_baseline_bias_experiments.ipynb`) are kept intentionally thin: they import
these scripts' own data-loading and compute functions — nothing is duplicated
— and only plot a couple of headline figures inline, for interactive
exploration. Run the scripts above for the full experiment battery. Each
notebook exposes its own `ERROR_MODEL` variable near the top of its
data-loading cell to select which `data/processed/` folder to read from;
figures/tables are keyed by `--error-model` the same way, under
`reports/{figures,tables}/<script>/<error_model>/`.

`run_phase4b_baseline_comparison.py` (and its notebook) score PHMM Forward
with the same length-corrected score used throughout Phase 4/4c (see
`scripts/evaluation/phmm_length_correction.py`): alpha is tuned on a 20%
held-out split of multimatch RF signals and applied to the other 80%. Every
model in that comparison — not just PHMM Forward — is restricted to the same
80% population (plus all single-match RF signals, which the correction can't
affect), so no method is compared on data another method's score was tuned
on. It also reports a random-choice baseline (`baseline_comparison.
random_choice_metrics`, simulated over 200 repetitions of picking a candidate
uniformly at random per RF signal) alongside the five scoring methods, as the
floor for "does matching beat chance at all".

`run_phase4c_open_set_evaluation.py`'s primary test for detecting dark
vessels is a **dark-vessel** split (`open_set.split_registry_dark_vessels` /
`build_dark_vessel_frame`): the vessels are split once by MMSI into a
known group (the only trajectories ever offered as candidates) and a
dark group (its RF detections are scored, but its trajectories are never candidates). This is
deliberately stricter than leave-one-out, which only ever excludes a query's
own track, so every other vessel stays a candidate and a leave-one-out
negative is not a faithful stand-in for a vessel without any AIS
trajectory. Dark-vessel negatives are built by
filtering the already-computed closed-set data (the prefilter and every
scoring method are pairwise, independent of which other candidates exist, so
this is equivalent to a fresh run restricted to the known group) — no leave-one-out
data is required, so it runs for either `--error-model` out of the box. Where
leave-one-out data *does* exist (built via
`scripts/modeling/AIS_RF_open_set/AIS_RF_leave_one_out.py`), the script also
reports it as a secondary comparison; where it doesn't, that section is
skipped with a message rather than failing the run.

## Repository structure

```bash
AIS_RF_matching_PHMM/
├─ config/
│  └─ mappings/vessel_type_names.json 
├─ data/                       (gitignored, generated by the steps below)
├─ notebooks/
│  ├─ 01_data_understanding.ipynb
│  ├─ 02_data_preparation.ipynb
│  ├─ 03_modeling.ipynb
│  ├─ 04_evaluation.ipynb                    (thin: see scripts/evaluation/run_phase4_evaluation.py)
│  ├─ 05_baseline_comparison.ipynb           (thin: see scripts/evaluation/run_phase4b_baseline_comparison.py)
│  ├─ 06_open_set_evaluation.ipynb           (thin: see scripts/evaluation/run_phase4c_open_set_evaluation.py)
│  └─ 07_baseline_bias_experiments.ipynb     (thin: see scripts/evaluation/run_phase4c_bias_experiments.py)
├─ reports/                    (generated by scripts/evaluation/run_phase4*.py, gitignored)
│  ├─ figures/<phase>/<error_model>/*.png
│  ├─ plot_data/<phase>/<error_model>.pkl   (figure inputs)
│  └─ tables/<phase>/<error_model>/*.csv
├─ scripts/
│  ├─ data_loading/     
│     └─ download_AIS_data.py        
│  ├─ data_preprocessing/
│     ├─ AIS/     
│        ├─ pre_filter_vessel_type.py       
│        ├─ step01_filter_short_tracks.py   
│        ├─ step02_filter_speed_outliers.py
│        ├─ step03_make_continuous_tracks.py   
│        └─ step04_create_sample_data.py 
│     ├─ RF/ 
│        └─ generate_synthethic_RF_data.py 
│  ├─ evaluation/
│     ├─ metrics.py                          (top-1 / ranking / significance metrics)
│     ├─ phmm_length_correction.py           (PHMM track-length bias correction)
│     ├─ bias_experiments.py                 (correct/incorrect/should-be split + feature builders)
│     ├─ baseline_comparison.py              (Phase 4b compute helpers)
│     ├─ open_set.py                         (Phase 4c dark-vessel compute helpers)
│     ├─ plots.py                            (every figure used by the run_phase4*.py scripts)
│     ├─ run_phase4_evaluation.py            (repeatable Phase 4 experiment runner)
│     ├─ run_phase4b_baseline_comparison.py  (repeatable Phase 4b runner)
│     ├─ run_phase4c_open_set_evaluation.py  (repeatable Phase 4c open-set runner)
│     ├─ run_phase4c_bias_experiments.py     (repeatable Phase 4c bias-experiment runner)
│     ├─ run_kappa_sensitivity.py            (kappa ablation)
│     ├─ generate_plots.py / regenerate_plots.py  (figure generation, redraw from reports/plot_data)
│  ├─ run_pipeline.py                         (one command: preselection to all reports)
│  ├─ modeling/     
│     ├─ AIS_RF_alignment/     
│        ├─ AIS_RF_forward_alignment.py       
│        ├─ AIS_RF_nn_baseline.py       
│        └─ forward.py 
│     ├─ AIS_RF_preselection/     
│        └─ AIS_RF_preselection.py
│     ├─ AIS_RF_open_set/
│        └─ AIS_RF_leave_one_out.py
│     ├─ hidden_states/     
│        └─ states.py  
│     ├─ transition_probabilities/     
│        ├─ AIS_RF_distribution.py    
│        ├─ match_distribution.py  
│        └─ transitions.py 
│     ├─ other/     
│        └─ viterbi.py
│  ├─ utils/     
│     └─ geo_utils.py         
│  ├─ visualization/   
│     ├─ map_AIS_RF_distribution.py   
│     ├─ map_both_distributions.py
│     ├─ map_match_distribution.py   
│     └─ map_vessel_routes.py 
├─ .gitignore
├─ .python-version
├─ README.md
├─ pyproject.toml
└─ uv.lock
```

