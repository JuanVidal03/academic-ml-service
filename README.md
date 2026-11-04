# Dropout Prediction ML Service

Machine learning service that estimates a university student's risk of dropping out, so
support staff can reach the right students before they leave. Built with FastAPI,
scikit-learn and SQLAlchemy.

This service is the backend for ths repo [Academic web service.](https://github.com/JuanVidal03/academic-web-service)

> **New here?** Read [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) — a step-by-step guide that
> names, for every step, the exact files and functions involved and how to verify the result.

---

## How it works, briefly

1. A historical dataset of 3,000 students is the training source (`data/`).
2. `scripts/train.py` compares five algorithms with cross-validation, tunes the winner,
   calibrates a decision threshold for recall, and saves `models/model.joblib`.
3. The API loads that model once at startup and serves predictions.
4. Every prediction is written to a PostgreSQL audit log — if a database is configured.

Training happens **offline**. The API never trains; it only loads and serves.

## Dataset

`data/dataset_desercion_academica_3000.csv`

| Property | Value |
|---|---|
| Students | 3,000 (one row each) |
| Dropout rate | 43.9% (1,317 dropped out, 1,683 retained) |
| Academic periods | 9 (2020-1 through 2024-1) |
| Programs | 10 |
| Features used | 34 — 13 categorical, 21 numeric |
| Missing values | Only `area_materia_critica` (1,507 students have no critical subject area) |

The classes are close to balanced, so no resampling or class weighting is applied. The
dropout rate is flat across all nine periods (0.41–0.49), so there is no temporal drift and
a stratified random split is a sound evaluation choice.

Three CSV columns are deliberately **not** features: `id_estudiante` and `periodo` (letting
the model key on them would be memorisation, not prediction) and `materia_mas_repetida`
(free text, superseded by `area_materia_critica`). `creditos_perdidos` was also dropped: it
equals `3 × materias_reprobadas` exactly, so it carries no additional information.

## Model performance

Measured on the 600-student held-out test set, at the calibrated threshold of **0.3505**:

| Metric | Value |
|---|---|
| ROC-AUC | 0.7695 |
| Recall | 0.7909 |
| Precision | 0.6029 |
| F1 | 0.6842 |
| Accuracy | 0.6800 |

Cross-validated comparison on the training set (5-fold, ranked by ROC-AUC):

| Model | ROC-AUC | F1 | Recall | Accuracy |
|---|---|---|---|---|
| **LogisticRegression** | **0.7741** | 0.6499 | 0.6186 | 0.7075 |
| SVM | 0.7590 | 0.6356 | 0.6053 | 0.6958 |
| GradientBoosting | 0.7573 | 0.6420 | 0.6205 | 0.6963 |
| RandomForest | 0.7571 | 0.6295 | 0.5996 | 0.6904 |
| HistGradientBoosting | 0.7330 | 0.6121 | 0.5873 | 0.6733 |

Logistic regression wins, which is a useful result rather than a disappointing one: the
relationship here is largely linear, so the simplest and most interpretable model is also
the most accurate, and the gradient-boosted alternatives were overfitting.

**Why recall is prioritised.** The errors are not symmetric. Missing a student who goes on
to drop out costs a person their degree; flagging a student who was fine costs one advising
conversation. The decision threshold is therefore calibrated to reach 80% recall rather than
left at the default 0.5, which raises recall from 0.58 to 0.79 — roughly 56 additional
at-risk students caught per 1,000 — at the cost of more false alarms. Tune this with
`TARGET_RECALL`.

Most influential features, by permutation importance (held-out ROC-AUC drop when shuffled):

| Feature | Importance |
|---|---|
| `nota_promedio` | 0.0622 |
| `porcentaje_asistencia` | 0.0348 |
| `semestres_repetidos` | 0.0176 |
| `materias_canceladas` | 0.0105 |
| `satisfaccion_con_programa` | 0.0040 |

Run `uv run scripts/analyze.py` to regenerate the full report.

> **Interpret with care.** ROC-AUC near 0.77 means the model is a useful prioritisation aid,
> not an oracle: roughly 40% of the students it flags would not have dropped out. Treat its
> output as a prompt for a human conversation, never as an automatic decision about a
> student.

---

## Quick start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL (optional — the service runs fully without it)

### Install

```bash
cd ml-service
uv sync
cp .env.example .env     # every value is optional; defaults work
```

### Train

```bash
uv run scripts/train.py          # ~7 seconds
```

Writes `models/model.joblib` and `models/metrics.json`.

### Serve

```bash
# Development (auto-reload)
uv run uvicorn dropout_service.main:app --reload --port 8000

# Production
uv run uvicorn dropout_service.main:app --host 0.0.0.0 --port 8000
```

Interactive API docs: <http://localhost:8000/docs>

### Analyse the data (optional)

```bash
uv run scripts/analyze.py        # writes reports/analysis.{md,json}
```

### Set up the database (optional)

Only needed to log predictions and use `GET /predictions`.

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/dropout_db"
uv run alembic upgrade head
```

A bare `postgresql://` URL is rewritten to `postgresql+psycopg://` automatically, so a URL
copied from Railway or Heroku works unchanged.

---

## API

Business endpoints live under `/api/v1`. `/health` is deliberately unversioned: it is an
operational contract that Railway's healthcheck points at, so it must never move.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness and readiness (used by Railway) |
| GET | `/` | Service banner |
| GET | `/api/v1/model/info` | Model metadata, threshold and test metrics |
| POST | `/api/v1/predict` | Assess one student |
| POST | `/api/v1/predict/batch` | Assess up to 100 students |
| GET | `/api/v1/predictions` | Paginated audit log (requires a database) |
| GET | `/docs` | Interactive OpenAPI documentation |

### Example

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "semestre": 3,
    "programa": "Ingenieria de Sistemas",
    "materias_matriculadas": 6,
    "materias_aprobadas": 4,
    "materias_reprobadas": 2,
    "materias_repetidas": 1,
    "materias_canceladas": 0,
    "semestres_repetidos": 0,
    "numero_materias_con_perdida_historica": 2,
    "nota_promedio": 3.2,
    "promedio_semestre_anterior": 3.0,
    "promedio_acumulado": 3.1,
    "area_materia_critica": "Matematicas y Ciencias Basicas",
    "porcentaje_asistencia": 85.0,
    "inasistencias": 5,
    "horas_estudio_semana": 10.0,
    "edad": 20,
    "genero": "M",
    "estrato": 3,
    "tipo_admision": "Regular",
    "modalidad": "Diurna",
    "distancia_a_universidad_km": 5.0,
    "trabaja": "No",
    "horas_trabajo_semana": 0,
    "ingreso_familiar": 2000000,
    "recibe_beca": "No",
    "beca_perdida_en_periodo": "No",
    "dificultad_economica": "No",
    "tiene_hijos": "No",
    "nivel_educativo_padres": "Secundaria",
    "apoyo_familiar_percibido": 4,
    "satisfaccion_con_programa": 4,
    "participacion_extracurricular": "No",
    "uso_tutorias": "Si"
  }'
```

Response (verified against the committed model):

```json
{
  "deserta": false,
  "probabilidad": 0.2289055258796182,
  "riesgo": "bajo"
}
```

- `deserta` — `true` when `probabilidad >= decision_threshold` (0.3505, not 0.5)
- `probabilidad` — estimated dropout probability
- `riesgo` — `bajo` (< 0.30), `medio` (0.30–0.60), `alto` (>= 0.60)

Risk bands are an **institutional policy choice**, not a statistical result. Change them
with `RISK_THRESHOLD_MEDIUM` and `RISK_THRESHOLD_HIGH`.

### Input contract

All 34 features are declared once, in `src/dropout_service/ml/schema.py`. Notes:

- `area_materia_critica` is **optional** — omit it or send `null` for the ~50% of students
  with no critical area. The pipeline supplies the fill value used during training.
- Categorical fields are validated against closed vocabularies, so a typo returns `422`
  instead of silently degrading the prediction.
- Unknown fields are rejected with `422`.

---

## Configuration

Every variable is optional. See `.env.example` for the full annotated list.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | unset | PostgreSQL connection. Unset = no prediction logging |
| `CORS_ORIGINS` | localhost:3000, :5173 | Browser origins allowed to call the API |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `TARGET_RECALL` | `0.80` | Recall the decision threshold must reach |
| `RISK_THRESHOLD_MEDIUM` | `0.30` | Lower bound of the `medio` band |
| `RISK_THRESHOLD_HIGH` | `0.60` | Lower bound of the `alto` band |
| `TEST_SIZE` | `0.2` | Held-out fraction |
| `CV_FOLDS` | `5` | Cross-validation folds |
| `RANDOM_STATE` | `42` | Seed, for reproducibility |

Never set `CORS_ORIGINS` to `*`: the API is called with credentials.

---

## Development

```bash
uv run pytest -v            # 78 tests
uv run ruff check .         # lint
uv run ruff format .        # format
```

### Three design rules worth knowing before you change anything

**1. Add a feature in exactly one place.** Every feature is a field on `StudentFeatures` in
`ml/schema.py`; the categorical and numeric lists are *derived* from it by introspection. The
previous version hand-wrote that list in five separate files, and two copies drifted apart —
training filled missing categories with `"Unknown"` while the API defaulted to `"N/A"`,
which silently corrupted predictions for half the student population.
`tests/ml/test_schema.py` fails if a field ever escapes the derivation.

**2. Anything that learns from data belongs inside the pipeline.** Imputation, scaling and
encoding all live in `ml/pipeline.py`, so scikit-learn refits them per cross-validation fold
and training and serving share one code path. Imputing in `ml/training/data.py` would leak
test-set statistics into every fold and let the two paths diverge.
`tests/ml/test_pipeline.py` and `tests/ml/test_no_train_serve_skew.py` guard both properties.

**3. The dependency arrows are tested, not just intended.** The serving path must never
import `ml/training/` — those modules read the CSV, which is excluded from the Docker image,
so an accidental import would pass every local test and fail only in production. And `ml/`
must never import FastAPI or the HTTP slices, so the model stays usable from a notebook or
a batch job. `tests/test_layering.py` enforces both.

A consequence worth knowing: `model.joblib` pickles a function from `ml/pipeline.py` by
import path, so **moving Python modules invalidates every previously trained model** with
`ModuleNotFoundError`. Retrain after any such refactor — it is deterministic and takes
seconds.

---

## License

MIT
