"""Loading the training dataset and splitting it.

Deliberately thin: no imputation, no scaling, no encoding. Anything that learns a statistic
from the data belongs in :mod:`dropout_service.ml.schema` so that cross-validation refits it per
fold. See that module for why.
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dropout_service.config import get_settings
from dropout_service.ml.schema import ALL_FEATURES, TARGET_COLUMN, TARGET_MAP

logger = logging.getLogger(__name__)


def prepare_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate a raw training frame and return ``(X, y)``.

    Takes a frame rather than reading a file, so this module stays free of any knowledge of
    where the data lives -- the database read is ``training_data/service.py``.

    ``X`` holds exactly the feature columns in :data:`ALL_FEATURES` order; ``y`` is the
    binary dropout label. Columns absent from the frame are created as all-missing so the
    pipeline's imputers handle them, rather than raising a ``KeyError`` here.

    Raises:
        ValueError: if the target column is absent or holds unmappable values.
    """
    df = frame.copy()
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Training data is missing the target column {TARGET_COLUMN!r}")

    y = df[TARGET_COLUMN]
    # The database stores the label as a boolean; the CSV stores "Si"/"No". bool is a
    # subclass of int, so {0, 1} covers True/False too.
    if y.dtype == bool or set(y.dropna().unique()) <= {0, 1}:
        y = y.astype(int)
    else:
        y = y.map(TARGET_MAP)
        if y.isna().any():
            unexpected = sorted(set(df.loc[y.isna(), TARGET_COLUMN].astype(str)))
            raise ValueError(f"Target has values outside {set(TARGET_MAP)}: {unexpected}")
        y = y.astype(int)

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        logger.warning(
            "Training data lacks %d feature column(s), treating as missing: %s",
            len(missing),
            missing,
        )
        for col in missing:
            df[col] = None

    logger.info("Prepared %d rows (dropout rate %.1f%%)", len(df), 100 * y.mean())
    return df[ALL_FEATURES], y


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
