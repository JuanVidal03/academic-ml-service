# Dropout Prediction ML Service

Machine learning service that estimates a university student's risk of dropping out, so
support staff can reach the right students before they leave. Built with FastAPI,
scikit-learn and SQLAlchemy.

This service is the backend for ths repo [Academic web service.](https://github.com/JuanVidal03/academic-web-service)

> **New here?** Read [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) — a step-by-step guide that
> names, for every step, the exact files and functions involved and how to verify the result.

---

## How it works, briefly

1. A historical dataset of 3,000 students is seeded from `data/` into PostgreSQL.
2. `scripts/train.py` compares five algorithms with cross-validation, tunes the winner,
   calibrates a decision threshold for recall, and stores the model in the database.
3. The API loads the active model version once at startup and serves predictions.
4. Every prediction is written to an audit log, linked to the model that produced it.

Training happens **offline**. The API never trains; it only loads and serves.

**PostgreSQL is required.** The model, the training data and the audit log all live in it,
so there is nothing to fall back to.

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

Measured on the 600-student held-out test set, at the calibrated threshold of **0.3025**
(`TARGET_RECALL=0.85`):

| Metric | Value | |
|---|---|---|
| Recall | **0.8631** | 86.3% of real dropouts are caught |
| Precision | 0.5835 | 58.4% of flagged students really dropped out |
| F1 | 0.6963 | |
| ROC-AUC | 0.7695 | threshold-independent; the ranking quality |
| Accuracy | 0.6700 | |

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
conversation. The decision threshold is therefore calibrated to reach 85% recall rather than
left at the default 0.5, which raises recall from 0.58 to 0.86 at the cost of more false
alarms: the service flags **65% of students to catch 86% of dropouts**, and about 42% of
those flags are precautionary. Tune this with `TARGET_RECALL`.

### Why ROC-AUC is 0.77 and will not become 0.85

Worth stating plainly, because "improve the model" usually means "add data" and here that
does not work. Two things were measured rather than assumed.

**More rows do not help.** The learning curve is flat:

| Training rows | Train ROC-AUC | Test ROC-AUC |
|---|---|---|
| 240 | 0.8571 | 0.7401 |
| 600 | 0.8271 | 0.7615 |
| 1,200 | 0.8088 | 0.7676 |
| 1,800 | 0.8028 | 0.7736 |
| 2,400 | 0.7934 | 0.7740 |

Quadrupling the data from 600 to 2,400 rows bought **+0.0125** ROC-AUC, and the last 600
rows bought **+0.0004**. Train and test scores have converged (0.793 vs 0.774), which is the
signature of a model limited by the *information in its features*, not by sample size. More
rows — real or synthetic — cannot close a 0.08 gap. Generating extra synthetic rows from the
same distribution would make the number look better only if they leaked into the test split,
which would be measuring nothing.

**Engineered features do not help either.** Six were tried — approval rate, grade trend,
gap against the cumulative average, work/study ratio, attendance×grade, and a composite
academic-risk count:

| Feature set | ROC-AUC |
|---|---|
| 34 baseline features | 0.7740 |
| + 6 engineered features | 0.7741 |
| + 6 engineered, HistGradientBoosting | 0.7479 |

**+0.0001.** The linear model already extracts what these columns contain; the ratios are
algebraic combinations of features it can already weight.

**What would actually move it:** different *columns*, not more rows — LMS login frequency,
assignment submission timing, financial holds, advising attendance. That is a data-collection
project, not a modelling one.

**What 85% is reachable on: recall.** Recall is a threshold choice, so `TARGET_RECALL=0.85`
reaches it directly — and F1 improved as a side effect (0.6842 → 0.6963). ROC-AUC is
unchanged at 0.7695, because moving a threshold cannot change how well a model *ranks*.
This is the honest way to put 85% on the dashboard; raising ROC-AUC to 0.85 with this
dataset is not available.

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
cp .env.example .env     # required: the service will not start without it
createdb dropout_db      # or point DATABASE_URL at an existing database
```

### Bring up the database

Three steps, in order. Each is idempotent.

```bash
uv run alembic upgrade head   # create the schema
uv run scripts/seed_db.py     # data/*.csv -> training_records + students
uv run scripts/train.py       # train -> model_versions (marked active)  ~7 seconds
```

`scripts/train.py` stores the fitted pipeline and its metrics as a new `model_versions`
row and makes it the active one. There are no model files.

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

### Create a user

Every endpoint except `/health` requires a signed-in user.

```bash
# A real account (prompts for a password, minimum 12 characters)
uv run scripts/create_user.py --email you@universidad.edu.co --role admin

# Or the local development account: test@universidad.edu.co / test123
uv run scripts/seed_dev_user.py --i-know-this-is-insecure
```

Roles are `adviser` (may request predictions) and `admin` (may also read the audit log).

A bare `postgresql://` URL is rewritten to `postgresql+psycopg://` automatically, so a URL
copied from a hosted provider works unchanged.

---

## API

Sign in at `POST /auth/login` and send the token as `Authorization: Bearer <token>`.
Accounts are created by an administrator; there is no public registration.

| Method | Endpoint | Access | Description |
|---|---|---|---|
| POST | `/auth/login` | open | Credentials → access token + refresh cookie |
| POST | `/auth/refresh` | cookie | Rotate the refresh token, issue a new access token |
| POST | `/auth/logout` | open | Revoke the refresh token and clear the cookie |
| GET | `/auth/me` | signed in | The current user |
| GET | `/health` | **open** | Liveness; the container's `HEALTHCHECK` calls it |
| GET | `/` | open | Service banner |
| GET | `/model/info` | signed in | Model metadata, threshold and test metrics |
| POST | `/predict` | signed in | Assess one student with the deployed model |
| POST | `/predict/compare` | signed in | Assess one student with **every** stored algorithm, best first |
| POST | `/predict/batch` | signed in | Assess up to 100 students with the deployed model |
| POST | `/predict/batch/compare` | signed in | Assess up to 100 students with **every** algorithm |
| GET | `/stats/overview` | signed in | Aggregate counts for the dashboard |
| GET | `/predictions` | **admin** | Paginated audit log, newest first |
| GET | `/docs` | open | Interactive OpenAPI documentation |

`/predictions` is admin-only because it returns every student's features and risk score; an
adviser working with a handful of students has no business reading all of it. An adviser
calling it gets **403**, not 401 — they are authenticated, just not permitted.
`/stats/overview` is open to any signed-in user because it returns counts, not records.

### Comparing algorithms

Training fits all five candidates and stores each one, so `/predict/compare` and
`/predict/batch/compare` can report what every algorithm would say about the same student. **Each applies its own calibrated
threshold** — the spread is real (0.2504 to 0.3800 in the current run), so reusing the
winner's cut-point would misrepresent the others. Two models can therefore report similar
probabilities and opposite verdicts, which is calibration rather than a bug.

The batch form costs the same five prediction passes as the single form, because each
candidate predicts the whole batch in one vectorised call rather than once per student.
Both write **one** audit row per student -- the deployed model's answer, the one that was
acted on -- not one per algorithm.

### Example

```bash
curl -X POST http://localhost:8000/predict \
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

- `deserta` — `true` when `probabilidad >= decision_threshold` (0.3025, not 0.5)
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

Configuration comes from `.env` only. The seven required variables have **no default in
the code**, so a missing one aborts startup with a `ValidationError` naming the field —
deliberate, since each one changes either who may call the API or which model gets trained.
See `.env.example` for the annotated list.

| Variable | Required | Purpose |
|---|---|---|
| `CORS_ORIGINS` | **yes** | Browser origins allowed to call the API |
| `JWT_SECRET` | **yes** | Access-token signing key, ≥32 chars. Never reuse or commit one |
| `ACCESS_TOKEN_MINUTES` | **yes** | Access-token lifetime (cannot be revoked once issued) |
| `REFRESH_TOKEN_DAYS` | **yes** | Refresh-token lifetime (revocable, rotated on use) |
| `COOKIE_SECURE` | no (`true`) | **Set `false` for local http**, or the browser drops the cookie |
| `COOKIE_SAMESITE` | no (`lax`) | `none` needs `COOKIE_SECURE=true`; only for cross-domain setups |
| `TEST_SIZE` | **yes** | Held-out fraction |
| `CV_FOLDS` | **yes** | Cross-validation folds |
| `RANDOM_STATE` | **yes** | Seed, for reproducibility |
| `TARGET_RECALL` | **yes** | Recall the decision threshold must reach |
| `RISK_THRESHOLD_MEDIUM` | **yes** | Lower bound of the `medio` band |
| `RISK_THRESHOLD_HIGH` | **yes** | Lower bound of the `alto` band |
| `DATABASE_URL` | **yes** | PostgreSQL connection |
| `LOG_LEVEL` | no (`INFO`) | Logging verbosity |

Never set `CORS_ORIGINS` to `*`: the API is called with credentials.

### Deploy

Build an image and run it anywhere; there is no platform-specific configuration.

```bash
docker build -t dropout-service:1.0.0 .
docker run -p 8000:8000 --env-file .env dropout-service:1.0.0

# push to any registry
docker tag dropout-service:1.0.0 <registry>/dropout-service:1.0.0
docker push <registry>/dropout-service:1.0.0
```

The image carries **neither** the training data nor a model: it reads the active model
version from the database at startup, so one image works against any environment. Run
`alembic upgrade head` before a new version serves traffic. Required settings are supplied
at run time (`--env-file`), never baked into the image.

---

## Development

```bash
uv run pytest -v            # 144 tests
uv run ruff check .         # lint
uv run ruff format .        # format
```

### Four design rules worth knowing before you change anything

**1. Add a feature in exactly one place.** Every feature is a field on `StudentFeatures` in
`ml/schema.py`; the categorical and numeric lists are *derived* from it by introspection. The
previous version hand-wrote that list in five separate files, and two copies drifted apart —
training filled missing categories with `"Unknown"` while the API defaulted to `"N/A"`,
which silently corrupted predictions for half the student population.
`tests/ml/test_schema.py` fails if a field ever escapes the derivation.

**2. Anything that learns from data belongs inside the pipeline.** Imputation, scaling and
encoding are built in `ml/schema.py` alongside the feature contract, so scikit-learn refits
them per cross-validation fold and training and serving share one code path. Imputing in
`ml/training/data.py` would leak test-set statistics into every fold and let the two paths
diverge. `tests/ml/test_preprocessing.py` and `tests/ml/test_no_train_serve_skew.py` guard
both properties.

**3. The dependency arrows are tested, not just intended.** The serving path must never
import `ml/training/` — those modules read the CSV, which is excluded from the Docker image,
so an accidental import would pass every local test and fail only in production. And `ml/`
must never import FastAPI or the HTTP slices, so the model stays usable from a notebook or
a batch job. `tests/test_layering.py` enforces both.

A consequence worth knowing: the stored artefact pickles a function from `ml/schema.py` by
import path, so **moving Python modules invalidates every previously trained model** with
`ModuleNotFoundError`. Retrain after any such refactor — it is deterministic and takes
seconds. It is also why the preprocessing lives in `ml/schema.py` and not under
`ml/training/`: loading a model must not drag training code into the API.

**4. The database holds the state; `ml/` holds none of it.** The artefact is a `BYTEA`
column and the metrics a `JSONB` one, because a fitted scikit-learn pipeline is *not*
JSON-serialisable — it carries numpy arrays, learned categories and scaler statistics.
`Predictor.from_bytes` takes bytes so the database read stays in
`model_registry/service.py`, which is what keeps `ml/` free of SQLAlchemy. Training
features are stored as one `JSONB` payload rather than 34 columns, for exactly the reason
rule 1 exists: a mirrored column list would be a second copy of the feature set, free to
drift.

---

## License

MIT
