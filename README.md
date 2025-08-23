# Pairing AIS with RF:  
# A Markovian Approach for Maritime Data Fusion

**Author:** Kyra Jongman  
**Program:** MSc Data Science in Business and Entrepreneurship  
**Date:** September 2025

## Objective

Maritime transport carries ~90% of global trade but faces congestion, safety risks, and illicit activities. Authorities rely on AIS for vessel tracking, yet AIS suffers from coverage gaps and manipulation vulnerabilities. Space-based RF detections offer a promising complementary source, but methods to fuse AIS and RF are underdeveloped. This project explores a Pair Hidden Markov Model (PHMM) as a proof of concept to link AIS with RF detections and improve vessel identification under normal operating conditions

## Repository structure

```bash
AIS_RF_matching_PHMM/
├─ config/
│  └─ mappings/vessel_type_names.json                     
├─ notebooks/
│  ├─ 01_data_understanding.ipynb      
│  ├─ 02_data_preparation.ipynb             
│  └─ 03_modeling.ipynb 
│  └─ 04_evaluation.ipynb       
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

### Step 2 — Create & sync the environment

```bash
uv sync --frozen
```

### Step 3 — (optional) Activate the virtualenv

```bash
# On macOS and Linux.
source .venv/bin/activate

# On Windows.
.\.venv\Scripts\Activate.ps1
```

### Step 4 — Reproduce the datasets

```bash
# Make data folders 
mkdir -p data/raw
mkdir -p data/processed

# 1) Download AIS data 
# Source: NOAA’s Marine Cadastre (https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/index.html)
# Run:
    uv run python scripts/download_AIS_data.py
# Output: data/raw/AIS_01_2024.pkl

# 2) Filter op cargotypes
# Run:  
    uv run python scripts/prefilter_vessel_type.py
# Output: data/processed/cargo_vessels.parquet

# 3) Preprocess data
# Run full '02_data_preparation.ipynb' notebook
    jupyter notebook notebooks/02_data_preparation.ipynb
# Output: 
    # 1. data/processed/AIS_sample_no_RF_5000.pkl
    # 2. data/processed/statistics_sample_5000.pkl
    # 3. data/processed/train_data_sample_5000.pkl

# 4) Preselect AIS–RF candidates 
# Run: 
    uv run python scripts/AIS_RF_preselection.py
# Output: data/processed/AIS_RF_preselection_data.pkl

# 4) Forward alignment scores (PHMM) 
# Run: 
    uv run python scripts/AIS_RF_forward_alignment.py
# Output: data/processed/AIS_RF_forward_scores_data.pkl
```

