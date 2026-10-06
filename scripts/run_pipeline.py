"""Run the full AIS-RF matching pipeline end to end.

Cleans previously generated pipeline outputs, then re-runs candidate
preselection, PHMM Forward + nearest-neighbor baseline scoring, the
leave-one-out open-set negatives, and every phase 4 evaluation report
(closed-set metrics, baseline comparison, dark-vessel / open-set
detection) for one or both RF position-error models ("uniform" and
"gaussian", see README.md). The PHMM-vs-NN misclassification-bias
significance experiments (phase 4c) are off by default and enabled
with --bias-experiments.

Does not regenerate the base sample data (AIS_sample_no_RF_5000.pkl,
statistics_sample_5000.pkl, train_data_sample_5000.pkl); that is
produced interactively by notebooks/02_data_preparation.ipynb.

--plots-only skips preselection/scoring/evaluation entirely and just
redraws every figure from the plot-input cache each evaluation step
already wrote (see scripts/evaluation/plot_cache.py and
scripts/evaluation/regenerate_plots.py) -- use it after tweaking
scripts/evaluation/plots.py, when nothing about the underlying scores
changed.

Usage:
    uv run python -m scripts.run_pipeline
    uv run python -m scripts.run_pipeline --error-model gaussian
    uv run python -m scripts.run_pipeline --bias-experiments
    uv run python -m scripts.run_pipeline --plots-only
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ERROR_MODELS = ("uniform", "gaussian")
NN_BASELINE_MODELS = ("euclidean", "haversine", "segment", "time_weighted")

MODELING_STEPS = (
    "scripts.modeling.AIS_RF_preselection.AIS_RF_preselection",
    "scripts.modeling.AIS_RF_alignment.AIS_RF_forward_alignment",
    "scripts.modeling.AIS_RF_alignment.AIS_RF_nn_baseline",
    "scripts.modeling.AIS_RF_open_set.AIS_RF_leave_one_out",
)

# (module, reports/{figures,tables} subfolder, gated behind
# --bias-experiments), in run order.
EVAL_PHASES = (
    ("scripts.evaluation.run_phase4_evaluation",
     "phase4_evaluation", False),
    ("scripts.evaluation.run_phase4b_baseline_comparison",
     "phase4b_baseline_comparison", False),
    ("scripts.evaluation.run_phase4c_open_set_evaluation",
     "phase4c_open_set_evaluation", False),
    ("scripts.evaluation.run_phase4c_bias_experiments",
     "phase4c_bias_experiments", True),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def data_dir(error_model: str) -> Path:
    """Return the data/processed folder for the given error model."""
    base = REPO_ROOT / "data" / "processed"
    return base if error_model == "uniform" else base / error_model


def check_base_data(error_model: str) -> None:
    """Fail fast if notebook 02's output is missing for this model."""
    base = REPO_ROOT / "data" / "processed"
    required = (
        base / "AIS_sample_no_RF_5000.pkl",
        base / "statistics_sample_5000.pkl",
        data_dir(error_model) / "train_data_sample_5000.pkl",
    )
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing base data for error model {error_model!r}: "
            f"{missing}. Run notebooks/02_data_preparation.ipynb "
            "first (see README.md)."
        )


def clean_outputs(error_model: str, phase_dirs: list[str]) -> None:
    """Delete previously generated pipeline outputs before recomputing.

    Only removes files this pipeline writes -- the base sample data
    (train_data_sample_5000.pkl and friends) is left untouched.
    """
    d = data_dir(error_model)
    filenames = [
        "AIS_RF_preselection_data.pkl",
        "AIS_RF_preselection_checkpoint.pkl",
        "true_match_prefilter_diagnostic.pkl",
        "sibling_candidate_overlap_diagnostic.pkl",
        "AIS_RF_forward_scores_data.pkl",
        "AIS_RF_forward_scores_checkpoint.pkl",
        "AIS_RF_preselection_leaveoneout_data.pkl",
        "AIS_RF_preselection_leaveoneout_checkpoint.pkl",
        "AIS_RF_forward_scores_leaveoneout_data.pkl",
        "AIS_RF_forward_scores_leaveoneout_checkpoint.pkl",
    ]
    for name in NN_BASELINE_MODELS:
        filenames += [
            f"AIS_RF_nn_baseline_{name}_scores_data.pkl",
            f"AIS_RF_nn_baseline_{name}_scores_checkpoint.pkl",
            f"AIS_RF_nn_baseline_{name}_scores_leaveoneout_data.pkl",
            f"AIS_RF_nn_baseline_{name}_scores_leaveoneout_checkpoint.pkl",
        ]

    removed = 0
    for name in filenames:
        path = d / name
        if path.exists():
            path.unlink()
            removed += 1

    for phase_dir in phase_dirs:
        for kind in ("figures", "tables"):
            path = REPO_ROOT / "reports" / kind / phase_dir / error_model
            if path.exists():
                shutil.rmtree(path)

    logger.info(
        "Cleaned %d cached file(s) and prior report dirs for %s",
        removed, error_model,
    )


def run_step(module: str, error_model: str) -> None:
    """Run one pipeline stage as `python -m <module> --error-model`."""
    cmd = [sys.executable, "-m", module, "--error-model", error_model]
    logger.info("-> %s", " ".join(cmd))
    started = time.monotonic()
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    logger.info(
        "<- %s done in %.1fs", module, time.monotonic() - started
    )


def run_pipeline(error_model: str, with_bias_experiments: bool, plots_only: bool = False) -> None:
    """Run every pipeline stage for a single RF error model."""
    logger.info("=== Error model: %s ===", error_model)

    if plots_only:
        run_step("scripts.evaluation.regenerate_plots", error_model)
        return

    check_base_data(error_model)

    phase_dirs = [
        phase_dir for _, phase_dir, gated in EVAL_PHASES
        if not gated or with_bias_experiments
    ]
    clean_outputs(error_model, phase_dirs)

    for module in MODELING_STEPS:
        run_step(module, error_model)

    for module, _, gated in EVAL_PHASES:
        if gated and not with_bias_experiments:
            continue
        run_step(module, error_model)


def main() -> None:
    """Parse CLI args and run the pipeline for the selected models."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the full AIS-RF matching pipeline (preselection, "
            "PHMM + NN baseline scoring, leave-one-out, and phase 4 "
            "evaluation/baseline-comparison/open-set reports) for "
            "one or both RF error models."
        ),
    )
    parser.add_argument(
        "--error-model", choices=[*ERROR_MODELS, "both"], default="both",
        help="RF bearing-error model(s) to run the pipeline for "
             "(default: both).",
    )
    parser.add_argument(
        "--bias-experiments", action="store_true",
        help="Also run the phase 4c PHMM-vs-NN misclassification-bias "
             "significance experiments (off by default).",
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Skip preselection/scoring/evaluation and just redraw "
             "every figure from cached plot inputs "
             "(scripts.evaluation.regenerate_plots).",
    )
    args = parser.parse_args()

    models = (
        list(ERROR_MODELS) if args.error_model == "both"
        else [args.error_model]
    )

    started = time.monotonic()
    for error_model in models:
        run_pipeline(error_model, args.bias_experiments, args.plots_only)
    logger.info(
        "Full pipeline done in %.1f minutes",
        (time.monotonic() - started) / 60,
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        logger.error("Pipeline step failed: %s", exc)
        sys.exit(exc.returncode)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
