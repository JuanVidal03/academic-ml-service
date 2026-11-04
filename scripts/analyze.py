#!/usr/bin/env python3
"""Generate the dataset analysis report.

Usage:
    uv run scripts/analyze.py
"""

from dropout_service.config import configure_logging, get_settings
from dropout_service.ml.training.analysis import run_analysis, save_analysis


def main() -> None:
    configure_logging()
    settings = get_settings()
    analysis = run_analysis()
    json_path, md_path = save_analysis(analysis, settings.reports_dir)

    info = analysis["dataset"]
    print("\n" + "=" * 62)
    print("Analysis complete")
    print("=" * 62)
    print(f"  Records      : {info['total_records']:,}")
    print(f"  Dropout rate : {info['dropout_rate']:.1%}")
    print("\n  Top 5 features by permutation importance:")
    for i, (name, vals) in enumerate(list(analysis["permutation_importance"].items())[:5], 1):
        print(f"    {i}. {name}: {vals['importance']:.4f}")
    print(f"\n  JSON    : {json_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
