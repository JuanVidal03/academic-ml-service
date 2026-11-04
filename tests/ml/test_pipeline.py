"""The pipeline must learn every statistic from training rows only."""

import numpy as np
import pandas as pd
import pytest

from dropout_service.ml.pipeline import build_preprocessor
from dropout_service.ml.schema import CATEGORICAL_FILL, NUMERIC_FEATURES


def _frame(values: list[float | None]) -> pd.DataFrame:
    """A frame whose every numeric column carries ``values``."""
    data = {col: list(values) for col in NUMERIC_FEATURES}
    return pd.DataFrame(data)


def test_median_imputation_is_fit_on_supplied_rows_only():
    """Leakage regression.

    The old code imputed medians over the whole dataset before splitting, so every CV fold
    saw statistics drawn partly from its own validation rows. With imputation inside the
    pipeline, the imputer must learn only from what it is fitted on.
    """
    preprocessor = build_preprocessor()
    train = _frame([1.0, 1.0, 1.0, None])
    for col in preprocessor.transformers[0][2]:
        train[col] = "Regular" if col == "tipo_admision" else "No"

    preprocessor.fit(train)
    numeric_imputer = preprocessor.named_transformers_["num"].named_steps["impute"]

    # Median of the training rows is 1.0. A leaky implementation that had also seen rows
    # of, say, 100.0 would produce something else.
    assert np.allclose(numeric_imputer.statistics_, 1.0)


@pytest.mark.parametrize("absent", [None, np.nan])
def test_missing_category_is_filled_with_the_shared_sentinel(absent):
    """Training and serving must agree on the fill value -- the Bug 1 guard.

    Parametrised over both spellings of "absent" on purpose: read_csv yields NaN during
    training while a Pydantic optional field yields None when serving. Both must land on
    the same category, or the two paths diverge again.
    """
    preprocessor = build_preprocessor()
    categorical_cols = preprocessor.transformers[0][2]

    frame = _frame([1.0, 2.0, 3.0])
    for col in categorical_cols:
        frame[col] = "Regular" if col == "tipo_admision" else "No"
    frame["area_materia_critica"] = [absent, "Salud", absent]

    preprocessor.fit(frame)
    encoder = preprocessor.named_transformers_["cat"].named_steps["encode"]
    index = list(categorical_cols).index("area_materia_critica")

    # OneHotEncoder sorts its categories, so compare as a set.
    assert set(encoder.categories_[index]) == {CATEGORICAL_FILL, "Salud"}


def test_none_and_nan_encode_identically():
    """A served None must produce the same feature row as a trained NaN."""
    preprocessor = build_preprocessor()
    categorical_cols = preprocessor.transformers[0][2]

    frame = _frame([1.0, 2.0, 3.0])
    for col in categorical_cols:
        frame[col] = "Regular" if col == "tipo_admision" else "No"
    frame["area_materia_critica"] = ["Salud", "Salud", "Salud"]
    preprocessor.fit(frame)

    with_none = frame.copy()
    with_none.loc[0, "area_materia_critica"] = None
    with_nan = frame.copy()
    with_nan.loc[0, "area_materia_critica"] = np.nan

    assert np.array_equal(preprocessor.transform(with_none), preprocessor.transform(with_nan))
