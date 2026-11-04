"""Regression test for the train/serve skew bug.

Before the refactor, ``clean_data`` filled missing categories with "Unknown" at training
time while the API schema defaulted ``area_materia_critica`` to "N/A". Because the encoder
was built with ``handle_unknown="ignore"``, the mismatch produced an all-zero one-hot row
instead of an error, silently changing the prediction for the ~50% of students who have no
critical subject area (0.2449 vs 0.2081 on the documented example).

The fix is structural: the schema carries ``None`` and the pipeline -- a single code path
shared by training and serving -- supplies the fill value. These tests pin that down.
"""

import pandas as pd
import pytest

from dropout_service.ml.schema import ALL_FEATURES, CATEGORICAL_FILL, StudentFeatures


def test_omitted_category_matches_the_value_used_at_training_time(predictor, student):
    """Omitting the field must be identical to the fill value the model was trained on."""
    del student["area_materia_critica"]
    served = predictor.predict([StudentFeatures(**student)])[0]

    # Feed the trained fill value straight through the fitted pipeline, bypassing the API
    # layer entirely. If the two paths agree, serving reproduces training exactly.
    row = StudentFeatures(**student).model_dump()
    row["area_materia_critica"] = CATEGORICAL_FILL
    frame = pd.DataFrame([row]).reindex(columns=ALL_FEATURES)
    expected = float(predictor.pipeline.predict_proba(frame)[0][1])

    assert served.probabilidad == pytest.approx(expected, abs=1e-12)


def test_the_fill_value_is_a_category_the_model_actually_learned(predictor):
    """Guards against the fill silently becoming an unseen, all-zero encoding."""
    preprocessor = predictor.pipeline.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["cat"].named_steps["encode"]
    columns = list(preprocessor.transformers_[0][2])
    categories = encoder.categories_[columns.index("area_materia_critica")]

    assert CATEGORICAL_FILL in list(categories)


def test_an_unseen_category_would_have_produced_a_different_answer(predictor, student):
    """Proves the test above has teeth.

    If an unrecognised spelling produced the same probability as the correct one, the first
    test could pass vacuously. It does not: a stale value genuinely changes the output,
    which is exactly why the original bug mattered.
    """
    del student["area_materia_critica"]
    correct = predictor.predict([StudentFeatures(**student)])[0].probabilidad

    row = StudentFeatures(**student).model_dump()
    row["area_materia_critica"] = "N/A"  # the old, wrong default
    frame = pd.DataFrame([row]).reindex(columns=ALL_FEATURES)
    stale = float(predictor.pipeline.predict_proba(frame)[0][1])

    assert correct != pytest.approx(stale, abs=1e-9)
