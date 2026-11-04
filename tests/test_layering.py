"""Architecture tests: the dependency arrows must point one way.

These run each import in a fresh subprocess, because once any test module has imported the
training code, ``sys.modules`` in this process can no longer tell us who pulled it in.
"""

import subprocess
import sys
import textwrap


def _assert_clean_import(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_serving_does_not_import_training_code():
    """Importing the app must not drag in the offline training modules.

    This matters in production, not just in theory: the training modules read the CSV from
    ``data/``, which is deliberately excluded from the Docker image. An accidental import
    would therefore pass every local test and fail only once deployed.
    """
    _assert_clean_import("""
        import sys
        import dropout_service.main  # noqa: F401

        leaked = sorted(
            m for m in sys.modules if m.startswith("dropout_service.ml.training")
        )
        assert not leaked, f"serving path imported training modules: {leaked}"
    """)


def test_ml_domain_does_not_import_its_http_consumer():
    """``ml/`` is consumed by ``predictions/``, never the other way around."""
    _assert_clean_import("""
        import sys
        import dropout_service.ml.predict  # noqa: F401

        leaked = sorted(
            m for m in sys.modules
            if m.startswith(("dropout_service.predictions", "dropout_service.system"))
            or m == "dropout_service.main"
        )
        assert not leaked, f"ml domain imported HTTP layer: {leaked}"
    """)


def test_ml_domain_does_not_import_fastapi():
    """Keeps the model usable from a notebook or a batch job, not just from the API."""
    _assert_clean_import("""
        import sys
        import dropout_service.ml.predict  # noqa: F401
        import dropout_service.ml.schema  # noqa: F401

        assert "fastapi" not in sys.modules, "ml domain pulled in fastapi"
    """)
