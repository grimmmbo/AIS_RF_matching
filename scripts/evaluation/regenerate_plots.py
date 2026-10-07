"""
Regenerate every phase 4 figure from the figure inputs the evaluation
scripts stored in reports/plot_data/, without recomputing the
metrics/scores that produced them.

Each phase's inputs are loaded and handed to the same
scripts.evaluation.generate_plots functions the run_phase4*.py scripts
call directly after computing, so a figure's title, axis labels, and
which data it is drawn from are defined exactly once (in
generate_plots.py).

Skips (with a message, not an error) any phase whose inputs are not on
disk yet. Run the corresponding scripts.evaluation.run_phase4*.py
script (or scripts.run_pipeline) first for those.

Usage:
    uv run python -m scripts.evaluation.regenerate_plots
    uv run python -m scripts.evaluation.regenerate_plots --error-model gaussian
"""
import argparse
from pathlib import Path

from scripts.evaluation import generate_plots
from scripts.evaluation.run_phase4b_baseline_comparison import MODEL_NAMES, RANDOM_BASELINE_NAME
from scripts.evaluation.run_phase4c_bias_experiments import EXPERIMENTS, LOGODDS_NAME, NN_NAME, PHMM_NAME
from scripts.run_pipeline import ERROR_MODELS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PHASES = {
    "phase4_evaluation": generate_plots.phase4_evaluation,
    "phase4b_baseline_comparison": lambda data, fig_dir: generate_plots.phase4b_baseline_comparison(
        data, fig_dir, MODEL_NAMES, RANDOM_BASELINE_NAME
    ),
    "phase4c_open_set_evaluation": generate_plots.phase4c_open_set_evaluation,
    "phase4c_bias_experiments": lambda data, fig_dir: generate_plots.phase4c_bias_experiments(
        data, fig_dir, EXPERIMENTS, [PHMM_NAME, LOGODDS_NAME, NN_NAME]
    ),
}


def main(error_models: list) -> None:
    """Redraw every phase's figures for the given error models."""
    total = 0
    for error_model in error_models:
        for phase_dir, draw in PHASES.items():
            table_dir = REPO_ROOT / "reports" / "tables" / phase_dir / error_model
            data = generate_plots.load_plot_data(table_dir)
            if data is None:
                print(f"skip {phase_dir}/{error_model}: no plot data yet (run its evaluation script first)")
                continue
            fig_dir = REPO_ROOT / "reports" / "figures" / phase_dir / error_model
            fig_dir.mkdir(parents=True, exist_ok=True)
            total += draw(data, fig_dir)
    print(f"Done: {total} figure(s) regenerated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate phase 4 figures from stored plot data, no recomputation."
    )
    parser.add_argument(
        "--error-model", choices=[*ERROR_MODELS, "both"], default="both",
        help="RF bearing-error model(s) to regenerate figures for (default: both).",
    )
    args = parser.parse_args()

    models = list(ERROR_MODELS) if args.error_model == "both" else [args.error_model]
    main(models)
