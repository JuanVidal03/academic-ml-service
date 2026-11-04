#!/usr/bin/env python3
"""Train the dropout prediction model.

Usage:
    uv run scripts/train.py
"""

from dropout_service.config import configure_logging
from dropout_service.ml.training.train import run_training


def main() -> None:
    configure_logging()
    result = run_training()
    metrics = result["test_metrics"]
    print("\n" + "=" * 62)
    print("Training complete")
    print("=" * 62)
    print(f"  Algorithm          : {result['best_model']}")
    print(f"  Decision threshold : {result['decision_threshold']:.4f}")
    print(f"  ROC-AUC            : {metrics['roc_auc']:.4f}")
    print(f"  Recall             : {metrics['recall']:.4f}")
    print(f"  Precision          : {metrics['precision']:.4f}")
    print(f"  F1                 : {metrics['f1']:.4f}")
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"\n  Model  : {result['model_path']}")
    print(f"  Metrics: {result['metrics_path']}")


if __name__ == "__main__":
    main()
