# Pairing AIS with RF:<br>A Markovian Approach for Maritime Data Fusion

**Author:** Kyra Jongman  
**Program:** MSc Data Science in Business and Entrepreneurship  
**Date:** September 2025

## Objective

Maritime transport carries around 90% of global trade but faces congestion, safety risks, and illicit activities. Authorities rely on Automatic Identification System (AIS) for vessel tracking, yet AIS suffers from coverage gaps and manipulation vulnerabilities. Space-based radio frequency (RF) detections offer a promising complementary source, but methods to fuse AIS and RF are underdeveloped. This project explores a Pair Hidden Markov Model (PHMM) as a proof of concept to link AIS with RF detections and improve vessel identification under normal operating conditions.

## Repository structure

```bash
AIS_RF_matching_PHMM/
├─ config/
│  └─ mappings/vessel_type_names.json 
├─ data/
│  └─ processed/AIS_RF_preselection_data.pkl                    
├─ notebooks/
│  ├─ 01_data_understanding.ipynb      
│  ├─ 02_data_preparation.ipynb             
│  └─ 03_modeling.ipynb 
│  └─ 04_evaluation.ipynb       
│  └─ 05_baseline_comparison.ipynb       
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
│  ├─ modeling/     
│     ├─ AIS_RF_alignment/     
│        ├─ AIS_RF_forward_alignment.py       
│        ├─ AIS_RF_nn_baseline.py       
│        └─ forward.py 
│     ├─ AIS_RF_preselection/     
│        └─ AIS_RF_preselection.py     
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
├─ main.py
├─ pyproject.toml
└─ uv.lock
```

## Instructions
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
    # 3. data/processed/train_data_sample_5000.pkl               (RF error model: uniform)
    # 4. data/processed/gaussian/train_data_sample_5000.pkl       (RF error model: gaussian)

# 4) Preselect AIS–RF candidates 
# Fast (seconds on the sample dataset, parallelized across all CPU cores).
# A precomputed result is included in the repo; to regenerate it yourself, run:
    uv run python scripts/modeling/AIS_RF_preselection/AIS_RF_preselection.py
# Output: data/processed/AIS_RF_preselection_data.pkl

# 4) Forward alignment scores (PHMM) 
# NOTE: On the full study dataset, this step is compute-intensive (tens of minutes).
# Progress is checkpointed per AIS track, so an interrupted run can be resumed
# by simply re-running the same command.
# Run: 
    uv run python scripts/modeling/AIS_RF_alignment/AIS_RF_forward_alignment.py
# Output: data/processed/AIS_RF_forward_scores_data.pkl
# Checkpoint: data/processed/AIS_RF_forward_scores_checkpoint.pkl

# 5) Nearest-neighbor baselines
# Runs four lower-bound baselines to compare the PHMM Forward alignment
# against: plain Euclidean, Haversine (great-circle), point-to-segment
# (perpendicular distance to the nearest AIS track leg), and time-weighted
# (point-to-segment distance combined with an interpolated time gap, using
# the same distance_threshold=6km/time_window_hours=3 defaults as the
# preselection step). Fast (seconds to ~2 minutes on the sample dataset);
# progress is checkpointed per AIS track like the Forward alignment step above.
# Run:
    uv run python scripts/modeling/AIS_RF_alignment/AIS_RF_nn_baseline.py
# Output:
    # data/processed/AIS_RF_nn_baseline_euclidean_scores_data.pkl
    # data/processed/AIS_RF_nn_baseline_haversine_scores_data.pkl
    # data/processed/AIS_RF_nn_baseline_segment_scores_data.pkl
    # data/processed/AIS_RF_nn_baseline_time_weighted_scores_data.pkl
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
  `Normal(0, sigma)` (default `sigma=2000` meters), so the resulting distance
  follows a Rayleigh distribution (mean ≈ 1.25 × sigma) and the bearing is
  uniform over the full circle.

The notebook generates both variants in the same run: `uniform` output keeps
the original flat file layout under `data/processed/` (backward compatible),
while `gaussian` output is written to its own `data/processed/gaussian/`
subfolder. Downstream scripts select which one to use via `--error-model`:

```bash
# Uses data/processed/ (default)
uv run python scripts/modeling/AIS_RF_preselection/AIS_RF_preselection.py --error-model uniform

# Uses data/processed/gaussian/
uv run python scripts/modeling/AIS_RF_preselection/AIS_RF_preselection.py --error-model gaussian
uv run python scripts/modeling/AIS_RF_alignment/AIS_RF_forward_alignment.py --error-model gaussian
uv run python scripts/modeling/AIS_RF_alignment/AIS_RF_nn_baseline.py --error-model gaussian
```

Each script writes its output into the same folder it read its input from, so
the two error models' pipeline artifacts never mix. `04_evaluation.ipynb` and
`05_baseline_comparison.ipynb` each expose an `ERROR_MODEL` variable near the
top of their data-loading cell to select which folder to read results from.



