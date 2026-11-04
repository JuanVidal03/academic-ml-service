"""Loading the training dataset and splitting it.

Deliberately thin: no imputation, no scaling, no encoding. Anything that learns a statistic
from the data belongs in :mod:`dropout_service.ml.pipeline` so that cross-validation refits it per
fold. See that module for why.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from dropout_service.config import get_settings
from dropout_service.ml.schema import ALL_FEATURES, TARGET_COLUMN, TARGET_MAP

logger = logging.getLogger(__name__)


def load_dataset(csv_path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Read the dataset and return ``(X, y)``.

    ``X`` holds exactly the feature columns in :data:`ALL_FEATURES` order; ``y`` is the
    binary dropout label. Columns absent from the CSV are created as all-missing so the
    pipeline's imputers handle them, rather than raising a ``KeyError`` here.

    Raises:
        FileNotFoundError: if the dataset is missing.
        ValueError: if the target column is absent or holds unmappable values.
    """
    path = csv_path or get_settings().csv_path
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset is missing the target column {TARGET_COLUMN!r}")

    y = df[TARGET_COLUMN].map(TARGET_MAP)
    if y.isna().any():
        unexpected = sorted(set(df.loc[y.isna(), TARGET_COLUMN].astype(str)))
        raise ValueError(f"Target column has values outside {set(TARGET_MAP)}: {unexpected}")

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        logger.warning(
            "Dataset lacks %d feature column(s), treating as missing: %s", len(missing), missing
        )
        for col in missing:
            df[col] = None

    logger.info("Loaded %d rows from %s (dropout rate %.1f%%)", len(df), path.name, 100 * y.mean())
    return df[ALL_FEATURES], y.astype(int)


def split_dataset(
    X: pd.DataFrame, y: pd.Series, test_size: float | None = None, random_state: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    A random split is appropriate for this dataset: the dropout rate is flat across all nine
    academic periods present (0.41-0.49), so there is no temporal drift for a time-based
    split to protect against.
    """
    settings = get_settings()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size if test_size is not None else settings.test_size,
        random_state=random_state if random_state is not None else settings.random_state,
        stratify=y,
    )
    logger.info("Split into %d train / %d test rows", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test
