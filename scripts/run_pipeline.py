"""Run the full AIS-RF matching pipeline end to end.

Cleans previously generated pipeline outputs, then re-runs candidate
preselection, PHMM Forward + nearest-neighbor baseline scoring, and
every phase 4 evaluation report needed for the paper (closed-set
metrics, baseline comparison, dark-vessel open-set detection,
misclassification-bias significance experiments, kappa sensitivity)
for the "gaussian" RF position-error model by default (see
README.md); pass --error-model uniform or --error-model both for the
other model(s).

The leave-one-out open-set negatives (a second, independent recompute
of preselection+scoring with each RF point's own track excluded) are
off by default -- enable with --leave-one-out. The dark-vessel/MMSI
open-set check always runs regardless, since it doesn't need the
leave-one-out data.

The kappa sensitivity ablation (scripts.evaluation.run_kappa_sensitivity)
needs forward scores for kappa=1,1 in addition to the default kappa=0.5,2
scores the main forward-alignment step already produces, so this runs
one extra forward-alignment pass with --kappa 1,1.

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
    uv run python -m scripts.run_pipeline --error-model both
    uv run python -m scripts.run_pipeline --leave-one-out
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

# kappa=(1, 1) used for the kappa-sensitivity ablation, in addition to
# the default kappa=(0.5, 2) the main forward-alignment step already
# produces -- see scripts.evaluation.run_kappa_sensitivity.
KAPPA_ABLATION = "1,1"

MODELING_STEPS = (
    "scripts.modeling.AIS_RF_preselection.AIS_RF_preselection",
    "scripts.modeling.AIS_RF_alignment.AIS_RF_forward_alignment",
    "scripts.modeling.AIS_RF_alignment.AIS_RF_nn_baseline",
)
LEAVE_ONE_OUT_STEP = "scripts.modeling.AIS_RF_open_set.AIS_RF_leave_one_out"

# (module, reports/{figures,tables} subfolder), in run order. All run
# by default; run_kappa_sensitivity needs run_phase4_evaluation's and
# run_phase4c_open_set_evaluation's output to validate its default-kappa
# row against, so it runs last.
EVAL_PHASES = (
    ("scripts.evaluation.run_phase4_evaluation",
     "phase4_evaluation"),
    ("scripts.evaluation.run_phase4b_baseline_comparison",
     "phase4b_baseline_comparison"),
    ("scripts.evaluation.run_phase4c_open_set_evaluation",
     "phase4c_open_set_evaluation"),
    ("scripts.evaluation.run_phase4c_bias_experiments",
     "phase4c_bias_experiments"),
    ("scripts.evaluation.run_kappa_sensitivity",
     "phase4_evaluation"),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def data_dir(error_model: str) -> Path:
    """Return the data/processed folder for the given error model."""
    return REPO_ROOT / "data" / "processed" / error_model


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
    from scripts.modeling.AIS_RF_alignment.AIS_RF_forward_alignment import kappa_cache_suffix

    kappa_ais, kappa_rf = (float(v) for v in KAPPA_ABLATION.split(","))
    ablation_suffix = kappa_cache_suffix(kappa_ais, kappa_rf)

    d = data_dir(error_model)
    filenames = [
        "AIS_RF_preselection_data.pkl",
        "AIS_RF_preselection_checkpoint.pkl",
        "true_match_prefilter_diagnostic.pkl",
        "sibling_candidate_overlap_diagnostic.pkl",
        "AIS_RF_forward_scores_data.pkl",
        "AIS_RF_forward_scores_checkpoint.pkl",
        f"AIS_RF_forward_scores_data{ablation_suffix}.pkl",
        f"AIS_RF_forward_scores_checkpoint{ablation_suffix}.pkl",
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


def run_step(module: str, error_model: str, extra_args: list[str] | None = None) -> None:
    """Run one pipeline stage as `python -m <module> --error-model [extra_args]`."""
    cmd = [sys.executable, "-m", module, "--error-model", error_model, *(extra_args or [])]
    logger.info("-> %s", " ".join(cmd))
    started = time.monotonic()
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    logger.info(
        "<- %s done in %.1fs", module, time.monotonic() - started
    )


def run_pipeline(error_model: str, with_leave_one_out: bool, plots_only: bool = False) -> None:
    """Run every pipeline stage for a single RF error model."""
    logger.info("=== Error model: %s ===", error_model)

    if plots_only:
        run_step("scripts.evaluation.regenerate_plots", error_model)
        return

    check_base_data(error_model)

    phase_dirs = list(dict.fromkeys(phase_dir for _, phase_dir in EVAL_PHASES))
    clean_outputs(error_model, phase_dirs)

    for module in MODELING_STEPS:
        run_step(module, error_model)

    # Extra forward-alignment pass for the kappa-sensitivity ablation
    # (default kappa=0.5,2 scores already came from the main pass above)
    run_step(
        "scripts.modeling.AIS_RF_alignment.AIS_RF_forward_alignment", error_model,
        extra_args=["--kappa", KAPPA_ABLATION],
    )

    if with_leave_one_out:
        run_step(LEAVE_ONE_OUT_STEP, error_model)

    for module, _ in EVAL_PHASES:
        if module == "scripts.evaluation.run_kappa_sensitivity":
            # Explicit, rather than relying on this matching the
            # module's own --kappas default -- keeps the two in sync
            # with KAPPA_ABLATION even if that default ever changes.
            run_step(module, error_model, extra_args=["--kappas", f"0.5,2;{KAPPA_ABLATION}"])
        else:
            run_step(module, error_model)


def main() -> None:
    """Parse CLI args and run the pipeline for the selected models."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the full AIS-RF matching pipeline (preselection, "
            "PHMM + NN baseline scoring, and every phase 4 "
            "evaluation/baseline-comparison/open-set/bias/kappa-"
            "sensitivity report needed for the paper) for one or more "
            "RF error models."
        ),
    )
    parser.add_argument(
        "--error-model", choices=[*ERROR_MODELS, "both"], default="gaussian",
        help="RF bearing-error model(s) to run the pipeline for "
             "(default: gaussian).",
    )
    parser.add_argument(
        "--leave-one-out", action="store_true",
        help="Also run the leave-one-out open-set negatives (a second, "
             "independent recompute of preselection+scoring with each RF "
             "point's own track excluded). Off by default -- the primary "
             "dark-vessel/MMSI open-set check always runs regardless.",
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
        run_pipeline(error_model, args.leave_one_out, args.plots_only)
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
