#!/usr/bin/env python3
"""Train the dropout prediction model and store it in the database.

Usage:
    uv run scripts/train.py
"""

from dropout_service.config import configure_logging, get_settings
from dropout_service.database import create_db_engine, create_session_factory
from dropout_service.ml.training.train import run_training
from dropout_service.model_registry.service import save_version
from dropout_service.training_data.service import load_training_frame


def main() -> None:
    configure_logging()
    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        frame = load_training_frame(session)
        result = run_training(frame)
        version = save_version(session, result.pipeline, result.metadata)
        version_id, version_label = version.id, version.version

    metrics = result.metadata["test_metrics"]
    print("\n" + "=" * 62)
    print("Training complete")
    print("=" * 62)
    print(f"  Algorithm          : {result.metadata['model_name']}")
    print(f"  Decision threshold : {result.metadata['decision_threshold']:.4f}")
    print(f"  ROC-AUC            : {metrics['roc_auc']:.4f}")
    print(f"  Recall             : {metrics['recall']:.4f}")
    print(f"  Precision          : {metrics['precision']:.4f}")
    print(f"  F1                 : {metrics['f1']:.4f}")
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"\n  Stored as model_versions id={version_id} ({version_label}), active")


if __name__ == "__main__":
    main()
