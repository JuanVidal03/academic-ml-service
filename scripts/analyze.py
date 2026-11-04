#!/usr/bin/env python3
"""Analyse the training data: a Markdown report for humans, a JSONB row for querying.

Usage:
    uv run scripts/analyze.py
"""

from dropout_service.config import configure_logging, get_settings
from dropout_service.database import create_db_engine, create_session_factory
from dropout_service.ml.training.analysis import render_markdown, run_analysis
from dropout_service.training_data.service import load_training_frame, save_analysis


def main() -> None:
    configure_logging()
    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        frame = load_training_frame(session)
        analysis = run_analysis(frame)
        report = save_analysis(session, analysis)
        report_id = report.id

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = settings.reports_dir / "analysis.md"
    md_path.write_text(render_markdown(analysis))

    info = analysis["dataset"]
    print("\n" + "=" * 62)
    print("Analysis complete")
    print("=" * 62)
    print(f"  Records      : {info['total_records']:,}")
    print(f"  Dropout rate : {info['dropout_rate']:.1%}")
    print("\n  Top 5 features by permutation importance:")
    for i, (name, vals) in enumerate(list(analysis["permutation_importance"].items())[:5], 1):
        print(f"    {i}. {name}: {vals['importance']:.4f}")
    print(f"\n  Stored as analysis_reports id={report_id}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
