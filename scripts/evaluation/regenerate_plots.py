"""
Regenerate every phase 4 figure from its already-computed CSV result
tables, without recomputing the metrics/scores that produced them.

Loads each table each run_phase4*.py script already writes to
reports/tables/ back into a DataFrame and hands it to the same
scripts.evaluation.generate_plots functions those scripts call
directly after computing -- so a figure's title, axis labels, and
which data it's drawn from are defined exactly once (in
generate_plots.py), never duplicated here.

Skips (with a message, not an error) any figure whose CSV(s) aren't on
disk yet -- run the corresponding scripts.evaluation.run_phase4*.py
script (or scripts.run_pipeline) first for those.

Usage:
    uv run python -m scripts.evaluation.regenerate_plots
    uv run python -m scripts.evaluation.regenerate_plots --error-model gaussian
"""
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from scripts.evaluation import generate_plots
from scripts.evaluation.run_phase4b_baseline_comparison import MODEL_NAMES, RANDOM_BASELINE_NAME
from scripts.evaluation.run_phase4c_bias_experiments import EXPERIMENTS, LOGODDS_NAME, NN_NAME, PHMM_NAME
from scripts.run_pipeline import ERROR_MODELS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _read(table_dir: Path, name: str, index_col: Optional[int] = None) -> Optional[pd.DataFrame]:
    path = table_dir / f"{name}.csv"
    if not path.exists():
        print(f"  skip {name}: {path} not found (run its evaluation script first)")
        return None
    # float_precision="round_trip": pandas' default C float parser is fast
    # but lossy (e.g. reads back "0.00011708916339792752" as
    # 0.0001170891633979) -- fine for rounded summary stats, but visibly
    # shifts the antialiased render of the dense ROC/PR curves
    return pd.read_csv(path, index_col=index_col, float_precision="round_trip")


def regenerate_phase4_evaluation(table_dir: Path, fig_dir: Path) -> int:
    data = {key: v for key in generate_plots.PHASE4_EVALUATION_KEYS if (v := _read(table_dir, key)) is not None}
    return generate_plots.phase4_evaluation(data, fig_dir)


def regenerate_phase4b_baseline_comparison(table_dir: Path, fig_dir: Path) -> int:
    data = {
        "metrics_all_candidates": _read(table_dir, "metrics_all_candidates", index_col=0),
        "metrics_multimatch_candidates": _read(table_dir, "metrics_multimatch_candidates", index_col=0),
        "score_margins_multimatch": _read(table_dir, "score_margins_multimatch"),
        "bootstrap_ci": _read(table_dir, "bootstrap_ci"),
    }
    data = {k: v for k, v in data.items() if v is not None}
    return generate_plots.phase4b_baseline_comparison(data, fig_dir, MODEL_NAMES, RANDOM_BASELINE_NAME)


def regenerate_phase4c_bias_experiments(table_dir: Path, fig_dir: Path) -> int:
    data = {}
    for slug, feature, *_ in EXPERIMENTS:
        key = f"experiment_{slug.split('.')[0]}_{feature}_points"
        value = _read(table_dir, key)
        if value is not None:
            data[key] = value
    return generate_plots.phase4c_bias_experiments(data, fig_dir, EXPERIMENTS, [PHMM_NAME, LOGODDS_NAME, NN_NAME])


def regenerate_phase4c_open_set_evaluation(table_dir: Path, fig_dir: Path) -> int:
    data = {}
    for slug, _title_suffix in generate_plots.OPEN_SET_SLUGS:
        for stem in (
            f"open_set_summary_{slug}", f"roc_curve_overall_{slug}",
            f"pr_curve_overall_{slug}", f"roc_curve_scored_only_{slug}",
        ):
            value = _read(table_dir, stem)
            if value is not None:
                data[stem] = value
    return generate_plots.phase4c_open_set_evaluation(data, fig_dir)


PHASES = [
    ("phase4_evaluation", regenerate_phase4_evaluation),
    ("phase4b_baseline_comparison", regenerate_phase4b_baseline_comparison),
    ("phase4c_open_set_evaluation", regenerate_phase4c_open_set_evaluation),
    ("phase4c_bias_experiments", regenerate_phase4c_bias_experiments),
]


def main(error_models: list) -> None:
    total = 0
    for error_model in error_models:
        print(f"=== Error model: {error_model} ===")
        for phase_dir, regenerate in PHASES:
            table_dir = REPO_ROOT / "reports" / "tables" / phase_dir / error_model
            fig_dir = REPO_ROOT / "reports" / "figures" / phase_dir / error_model
            if not table_dir.exists():
                print(f"  skip {phase_dir}/{error_model}: no result tables yet at {table_dir}")
                continue
            fig_dir.mkdir(parents=True, exist_ok=True)
            n = regenerate(table_dir, fig_dir)
            print(f"  {phase_dir}/{error_model}: wrote {n} figure(s) to {fig_dir}")
            total += n
    print(f"\nDone -- {total} figure(s) regenerated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate phase 4 figures from saved result tables, no recomputation."
    )
    parser.add_argument(
        "--error-model", choices=[*ERROR_MODELS, "both"], default="both",
        help="RF bearing-error model(s) to regenerate figures for (default: both).",
    )
    args = parser.parse_args()

    models = list(ERROR_MODELS) if args.error_model == "both" else [args.error_model]
    main(models)
