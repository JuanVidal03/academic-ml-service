# Walkthrough: how this service works, step by step

A practical guide to the dropout-prediction service. Every step states the command, the
**sequence of files and functions** it executes in order, what to expect, and how to verify
it worked.

Written against the code as it stands; every command here was run and its output checked.

**Contents**

1. [Architecture in one picture](#1-architecture-in-one-picture)
2. [Set up](#2-set-up)
3. [Analyse the dataset](#3-analyse-the-dataset)
4. [Train the model](#4-train-the-model)
5. [Run the API](#5-run-the-api)
6. [Anatomy of a prediction request](#6-anatomy-of-a-prediction-request)
7. [Test it](#7-test-it)
8. [The database](#8-the-database)
9. [Deploy](#9-deploy)
10. [Authentication (planned)](#10-authentication-planned)
11. [Troubleshooting](#11-troubleshooting)
12. [Design decisions](#12-design-decisions-and-why)

---

## 1. Architecture in one picture

There are two separate flows. They share the preprocessing pipeline and nothing else.

```
OFFLINE (seeding, once) ─ data/*.csv ─> training_data/service.py:seed_from_csv
                                        └─> ml.training_records + ml.students

OFFLINE (training) ─ run by a developer, seconds
  ml.training_records
    └─> training_data/service.py:load_training_frame   (JSONB -> DataFrame)
          └─> ml/training/data.py:prepare_dataset ─> split_dataset
                └─> ml/schema.py:build_pipeline
                      └─> ml/training/train.py:compare_models ─> tune_model
                            ─> select_threshold ─> evaluate
                            └─> model_registry/service.py:save_version
                                  └─> ml.model_versions (BYTEA + JSONB, is_active)

STARTUP ─ once
  main.py:lifespan ─> model_registry/service.py:load_active
                       └─> ml.model_versions ─> ml/predict.py:Predictor.from_bytes

ONLINE (serving) ─ runs per HTTP request, milliseconds
  HTTP POST /predict
    └─> main.py            (app, CORS, error handlers)
    └─> predictions/router.py:predict
          └─> ml/schema.py:StudentFeatures      (validate the request)
          └─> ml/predict.py:Predictor.predict   (reuses the SAME pipeline, now fitted)
                └─> PredictionResult ─> JSON response
          └─> predictions/service.py:log_predictions   (background, after the response)
                └─> ml.predictions, linked to the model version and (optionally)
                    the student
```

The model is loaded **once** at startup, not per request. The arrow worth remembering is
that both flows pass through `ml/schema.py`: that is what guarantees a student is
preprocessed identically whether they are a training row or a live request.

### How the code is organised

Grouped by **feature**, not by technical layer. Each slice owns its router, its schemas,
its tables and its logic, so adding a capability means adding a directory rather than
editing four shared files.

```
src/dropout_service/
  main.py            create_app() + the ASGI app      <- uvicorn entrypoint
  config.py          Settings (all required, from .env), configure_logging
  database.py        engine, session factory, Base    (no table definitions)
  dependencies.py    shared Depends providers + Annotated aliases

  ml/                the model — a domain library: no HTTP, no FastAPI, no SQLAlchemy
    schema.py        THE single source of truth: the 34 features AND the
                     preprocessing pipeline that transforms them
    predict.py       Predictor (from_bytes/dump), PredictionResult   (serving)
    training/        offline only — never imported by the API
      data.py        prepare_dataset, split_dataset
      train.py       compare, tune, calibrate  (persists nothing)
      analysis.py    exploratory reports

  model_registry/    the trained artefact, stored in the database
    models.py        ml.model_versions (artifact BYTEA, metrics JSONB)
    service.py       save_version(), load_active()

  training_data/     the dataset and the analysis reports
    models.py        ml.training_records (features JSONB), ml.analysis_reports
    service.py       seed_from_csv(), load_training_frame(), save_analysis()

  students/          ml.students — identity, so a prediction ties to a person
  users/             ml.users — storage for the authentication work (§10)

  predictions/       feature slice
    router.py        /predict, /predict/batch, /predictions
    schemas.py       request/response wrappers, MAX_BATCH_SIZE
    models.py        the ml.predictions table
    service.py       log_predictions(), fetch_predictions_page()

  system/            ops + deployed-model metadata
    router.py        /health, /, /model/info

  auth/              not created yet — the slot §10 fills (the table already exists)
```

Everything sits under one root package on purpose, rather than as flat `src/ml/`,
`src/predictions/`, `src/config.py` directly under `src/`. That keeps the project to a
single, unique top-level import name and keeps it cleanly installable —
[§12](#12-design-decisions-and-why) sets out the reasoning and what it costs.

**The dependency rule, which is enforced by `tests/test_layering.py`:**

```
system/ ─┐
         ├─> ml/  ───> ml/training/   NO. Serving must never import training.
predictions/ ─┘
         ml/ ───> predictions/        NO. The domain must not know its HTTP consumer.
         ml/ ───> fastapi             NO. The model stays usable from a notebook.
         ml/ ───> sqlalchemy          NO. from_bytes takes bytes; the registry reads.
```

`ml/` is deliberately **not** a feature slice: it has no HTTP surface and is consumed *by*
`predictions/`. Giving it a `router.py` would invert that relationship.

`/model/info` lives in `system/` rather than `predictions/` because it describes the
deployed artefact — version, threshold, metrics — which is operational introspection
rather than a prediction.

### URL layout

Flat, unversioned. There will be no `v2`, so a prefix would be indirection that buys
nothing; the two slice routers are included directly in `main.py`.

```
GET  /health          GET  /model/info      POST /predict
GET  /                GET  /predictions     POST /predict/batch
GET  /docs            GET  /openapi.json
```

`/health` is the one path with an external contract: the container's `HEALTHCHECK` calls
it.

## 2. Set up

**Commands**

```bash
cd ml-service
uv sync
cp .env.example .env
```

**File sequence**

1. `pyproject.toml` — `uv` reads dependencies and the `[tool.pytest]` / `[tool.ruff]` config
2. `uv.lock` — exact pinned versions, so every machine gets the same build
3. `.env.example` → `.env` — your local configuration

A database is required too. The full bring-up, in order — each step idempotent:

```bash
createdb dropout_db             # or point DATABASE_URL at an existing one
uv run alembic upgrade head     # create the ml schema
uv run scripts/seed_db.py       # data/*.csv -> training_records + students
uv run scripts/train.py         # train -> model_versions (marked active)
```

Until `scripts/train.py` has run there is no active model version, so the service starts
`degraded` and `/predict` returns 503. That is reported, not fatal.

`cp .env.example .env` is **required**, not optional. The seven tuneable settings have no
default in the code, so a missing one aborts startup with a `ValidationError` naming the
field. `.env.example` ships the intended values, so copying it is enough. Only
`DATABASE_URL` and `LOG_LEVEL` are optional.

**Verify**

```bash
uv run python -c "import dropout_service; print(dropout_service.__version__)"
# 1.0.0
```

---

## 3. Analyse the dataset

Optional, but it is how you learn what the data actually contains.

**Command**

```bash
uv run scripts/analyze.py
```

**File sequence**

| # | File : function | What it does |
|---|---|---|
| 1 | `scripts/analyze.py:main` | Entry point |
| 2 | `config.py:configure_logging` | Set up log format and level |
| 3 | `config.py:get_settings` | Resolve `data_dir`, `reports_dir` |
| 4 | `training_data/service.py:load_training_frame` | `SELECT features, deserto`; JSONB → DataFrame |
| 5 | `ml/training/analysis.py:run_analysis` | Orchestrates everything below |
| 6 | `ml/training/data.py:prepare_dataset` | Validate the frame, return `(X, y)` |
| 6 | `analysis.py:describe_dataset` | Row counts, dropout rate, feature counts |
| 7 | `analysis.py:summarise_features` | Per-feature mean/std/min/max, or value counts |
| 8 | `analysis.py:dropout_rate_by_category` | Dropout rate within each category level |
| 9 | `analysis.py:numeric_correlations` | Pearson correlation with the target |
| 10 | `analysis.py:mutual_information` | MI per feature (categoricals flagged discrete) |
| 11 | `analysis.py:feature_importance` | Permutation importance via `ml/schema.py` |
| 12 | `analysis.py:render_markdown` | Write `reports/analysis.md` for humans |
| 13 | `training_data/service.py:save_analysis` | Store the structured result in `ml.analysis_reports` |

**Expect** a few seconds, and:

```
  Records      : 3,000
  Dropout rate : 43.9%

  Top 5 features by permutation importance:
    1. nota_promedio: 0.0622
    2. porcentaje_asistencia: 0.0348
    3. semestres_repetidos: 0.0176
    4. materias_canceladas: 0.0105
    5. satisfaccion_con_programa: 0.0040
```

**Verify**

```bash
head -40 reports/analysis.md
```

**Why two different importance measures?** They answer different questions. Mutual
information asks "does this feature carry information about the outcome at all?", including
non-linear relationships, one feature at a time. Permutation importance asks "does *this
model* actually lose accuracy on unseen data without it?" — so it accounts for the model and
for features that are redundant with one another. Permutation importance is the one to trust
when deciding what matters in practice; MI is useful for spotting signal the current model
is failing to exploit.

---

## 4. Train the model

**Command**

```bash
uv run scripts/train.py
```

**File sequence**

| # | File : function | What it does |
|---|---|---|
| 1 | `scripts/train.py:main` | Entry point |
| 2 | `config.py:configure_logging` | Logging |
| 3 | `training_data/service.py:load_training_frame` | Read `ml.training_records`; JSONB → DataFrame |
| 4 | `ml/training/train.py:run_training` | Orchestrates everything below |
| 5 | `ml/training/data.py:prepare_dataset` | Validate the frame; raises on a corrupt target |
| 6 | `ml/training/data.py:split_dataset` | Stratified 80/20 → 2,400 train / 600 test |
| 7 | `train.py:compare_models` | For each of 5 algorithms: `_seeded` → `ml/schema.py:build_pipeline` → `cross_validate` with 4 scorers |
| 8 | `train.py:tune_model` | `GridSearchCV` over `PARAM_GRIDS[winner]`, refits the best |
| 9 | `train.py:select_threshold` | `cross_val_predict` out-of-fold → `precision_recall_curve` → highest threshold meeting `TARGET_RECALL` |
| 10 | `train.py:evaluate` | Test-set metrics **at that threshold** |
| 11 | `model_registry/service.py:save_version` | `joblib` → `BYTEA`, metadata → `JSONB`, previous version deactivated in the same transaction |

Inside step 7, `ml/schema.py:build_pipeline` → `build_preprocessor` assembles:

- categorical: `coerce_missing_to_nan` → `SimpleImputer(constant, "Unknown")` → `OneHotEncoder`
- numeric: `SimpleImputer(median)` → `StandardScaler`

**Expect** under ten seconds, ending with:

```
  Algorithm          : LogisticRegression
  Decision threshold : 0.3505
  ROC-AUC            : 0.7695
  Recall             : 0.7909
  Precision          : 0.6029
  F1                 : 0.6842
  Accuracy           : 0.6800
```

**Verify**

```bash
psql "$DATABASE_URL" -tAc "
  select algorithm, n_features, round(decision_threshold::numeric, 4),
         round((metrics->'test_metrics'->>'roc_auc')::numeric, 4), length(artifact)
  from ml.model_versions where is_active;"
# LogisticRegression|34|0.3505|0.7695|10874
```

**Sanity check, not celebration.** ROC-AUC should land near **0.77**. A sudden jump to 0.95+
almost always means leakage (a feature that encodes the answer) rather than a better model —
investigate before shipping it.

**Two things here are easy to get wrong, and both are deliberate:**

*The threshold is chosen on out-of-fold training predictions, never on the test set.* Picking
it on the test set would tune the operating point to the very rows used to report
performance, making the reported recall optimistic. Step 9 keeps the test set untouched
until step 10.

*The threshold is 0.35, not 0.5.* `pipeline.predict()` hardcodes 0.5, which is why
`Predictor` calls `predict_proba` and compares against the stored threshold instead. Raising
recall from 0.58 to 0.79 is the entire point of calibrating it.

**The artefact is coupled to the package path.** The stored artefact is a pickle, and it holds
`ml/schema.py:coerce_missing_to_nan` by its *import path*. Renaming or moving the package
therefore makes every previously trained model unloadable with
`ModuleNotFoundError`. Retraining is the fix (it is deterministic and takes seconds), but
in production this means a release that moves Python modules must ship a freshly trained
artefact — the model is not portable across refactors.

---

## 5. Run the API

**Command**

```bash
uv run uvicorn dropout_service.main:app --reload --port 8000
```

**File sequence at startup**

| # | File : function | What it does |
|---|---|---|
| 1 | `main.py:create_app` | Build the `FastAPI` instance |
| 2 | `config.py:get_settings` | Read environment |
| 3 | `main.py:lifespan` | Startup hook (runs once) |
| 4 | `config.py:configure_logging` | Logging |
| 5 | `database.py:create_db_engine` / `create_session_factory` | Connect |
| 6 | `model_registry/service.py:load_active` → `Predictor.from_bytes` | Load the active model version into memory |
| 7 | `CORSMiddleware` | Restrict browser origins to `CORS_ORIGINS` |
| 8 | `main.py` exception handler | Map `SQLAlchemyError` → 503 |
| 9 | `system/router.py:router` | Mount `/health`, `/` and `/model/info` |
| 10 | `predictions/router.py:router` | Mount the prediction endpoints |

Resources land on `app.state` and reach routes through `dependencies.py`. If the database is
unreachable or there is no active model version, startup logs it and continues — `/health`
reports `degraded` and prediction endpoints return `503`. That tolerance is deliberate:
crash-looping through a brief outage would hide the reason.

**Verify**

```bash
curl -s localhost:8000/health
# {"status":"healthy","model_loaded":true,"database":{"connected":true}}

curl -s localhost:8000/
# {"service":"dropout-prediction-api","docs":"/docs","health":"/health"}
```

`model_loaded: true` means an active `model_versions` row was found. Then open
<http://localhost:8000/docs>.

---

## 6. Anatomy of a prediction request

What happens on `POST /predict`, in order.

| # | File : function | What it does |
|---|---|---|
| 1 | `predictions/router.py` | Match `/predict` |
| 2 | `ml/schema.py:StudentFeatures` | Validate the body: ranges, closed vocabularies, no unknown fields. Invalid → `422`, and the handler never runs |
| 3 | `dependencies.py:get_predictor` | Hand over the in-memory `Predictor`; `503` if absent |
| 4 | `dependencies.py:get_session_factory` | Session factory, or `None` without a database |
| 5 | `predictions/router.py:predict` | The handler — plumbing only |
| 6 | `ml/predict.py:Predictor.predict` | Build a one-row DataFrame, `reindex` to `ALL_FEATURES` order |
| 7 | `ml/schema.py` steps (fitted) | `coerce_missing_to_nan` → impute → encode → scale |
| 8 | sklearn estimator | `predict_proba` → dropout probability |
| 9 | `ml/predict.py:Predictor.risk_level` | Band it: `bajo` / `medio` / `alto` |
| 10 | `ml/predict.py:PredictionResult` | Serialise `{deserta, probabilidad, riesgo}` |
| 11 | `predictions/service.py:log_predictions` | **After** the response: one `add_all` + one `commit` |

Step 11 runs in a `BackgroundTasks` callback, so database latency never delays a prediction
and a logging failure is recorded in the logs but never fails the request — the prediction
already succeeded. It lives in `service.py` rather than the router so it can be tested
without an HTTP request (`tests/predictions/test_service.py`).

`POST /predict/batch` is the same path with a list. `Predictor.predict` always takes
a list; a single prediction is `predict([student])[0]`, so single and batch requests cannot
diverge in behaviour.

### Try it

```bash
# Save a request body
cat > /tmp/student.json <<'JSON'
{"semestre":3,"programa":"Ingenieria de Sistemas","materias_matriculadas":6,
"materias_aprobadas":4,"materias_reprobadas":2,"materias_repetidas":1,
"materias_canceladas":0,"semestres_repetidos":0,
"numero_materias_con_perdida_historica":2,"nota_promedio":3.2,
"promedio_semestre_anterior":3.0,"promedio_acumulado":3.1,
"area_materia_critica":"Matematicas y Ciencias Basicas","porcentaje_asistencia":85.0,
"inasistencias":5,"horas_estudio_semana":10.0,"edad":20,"genero":"M","estrato":3,
"tipo_admision":"Regular","modalidad":"Diurna","distancia_a_universidad_km":5.0,
"trabaja":"No","horas_trabajo_semana":0,"ingreso_familiar":2000000,"recibe_beca":"No",
"beca_perdida_en_periodo":"No","dificultad_economica":"No","tiene_hijos":"No",
"nivel_educativo_padres":"Secundaria","apoyo_familiar_percibido":4,
"satisfaccion_con_programa":4,"participacion_extracurricular":"No","uso_tutorias":"Si"}
JSON

curl -s -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' -d @/tmp/student.json
```

Expected, exactly:

```json
{"deserta":false,"probabilidad":0.2289055258796182,"riesgo":"bajo"}
```

Now check the model responds to the signal — drop the grades and attendance:

```bash
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d "$(python3 -c "
import json; d=json.load(open('/tmp/student.json'))
d.update(nota_promedio=1.3, porcentaje_asistencia=42.0, promedio_acumulado=1.5,
         materias_reprobadas=4, inasistencias=22, semestres_repetidos=2)
print(json.dumps(d))")"
# {"deserta":true,"probabilidad":0.958...,"riesgo":"alto"}
```

### The verification that matters most

About half of all students have no critical subject area. Omitting the field and sending it
as `null` must give **byte-identical** answers, and both must match what the model saw during
training. This is the regression check for the bug described in
[§12](#12-design-decisions-and-why).

```bash
python3 -c "
import json; d=json.load(open('/tmp/student.json')); d.pop('area_materia_critica')
json.dump(d, open('/tmp/omitted.json','w'))
d['area_materia_critica']=None; json.dump(d, open('/tmp/null.json','w'))"

curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @/tmp/omitted.json
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @/tmp/null.json
```

Both must print `0.2186075960472946`. If they differ, the train/serve contract is broken.

And a stale `"N/A"` — the value an older version of this API defaulted to — must now be
*rejected* rather than silently mis-encoded:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d "$(python3 -c "
import json; d=json.load(open('/tmp/student.json'))
d['area_materia_critica']='N/A'; print(json.dumps(d))")"
# 422
```

### Every endpoint, by hand

```bash
curl -s localhost:8000/health
curl -s localhost:8000/model/info | python3 -m json.tool | head -20
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @/tmp/student.json
curl -s -X POST localhost:8000/predict/batch -H 'Content-Type: application/json' \
  -d "{\"students\":[$(cat /tmp/student.json),$(cat /tmp/student.json)]}"
curl -s localhost:8000/predictions            # 503 without a database
```

Expected failure modes:

| Request | Status | Why |
|---|---|---|
| `semestre: -1` | 422 | Below the allowed minimum |
| `genero: "Masculino"` | 422 | Not in the closed vocabulary (`F`/`M`/`Otro`) |
| `{"inventado": 1}` added | 422 | Unknown fields are rejected |
| `nota_promedio` omitted | 422 | Required |
| 101 students in a batch | 422 | Cap is 100 |
| `/predictions` with no DB | 503 | Needs PostgreSQL |
| `/predictions` with an unreachable DB | 503 | Mapped from `SQLAlchemyError` in `main.py`, so it is never an opaque 500 |
| `POST /records` | 404 | Removed — training data comes from the CSV |

---

## 7. Test it

**Command**

```bash
uv run pytest -v
```

**File sequence**

1. `pyproject.toml` `[tool.pytest.ini_options]` — `pythonpath = ["src"]`, `FutureWarning`
   from our own package treated as an error
2. `tests/conftest.py` — fixtures, which run first:
   - `raw_frame` / `dataset` → the CSV read directly, standing in for a database read
   - `model_artifact` → trains a small `LogisticRegression` and returns joblib **bytes**,
     which is what a `model_versions` row holds
   - `test_settings` → `Settings` pointing at an unreachable database
   - `predictor` → `ml/predict.py:Predictor.from_bytes`
   - `fake_session` → records what it was asked to persist; never opens a connection
   - `client` → `main.py:create_app` in a `TestClient` (runs the real lifespan)
   - `student` → a valid request body
3. The test modules, mirroring the package layout:

| File | Covers |
|---|---|
| `tests/ml/test_schema.py` | Feature lists match the contract; invalid input rejected |
| `tests/ml/test_preprocessing.py` | **Imputation fit on train rows only (leakage guard)**; `None` ≡ `NaN` |
| `tests/ml/test_no_train_serve_skew.py` | **Serving reproduces training exactly** |
| `tests/ml/test_predict.py` | Probability bounds, determinism, threshold, risk bands |
| `tests/ml/training/test_data.py` | Loading, target mapping, excluded columns, split properties |
| `tests/ml/training/test_train.py` | Grid keys valid, threshold hits target, end-to-end run |
| `tests/predictions/test_router.py` | Status codes, response shape, batch order, validation, prefix enforcement |
| `tests/predictions/test_service.py` | `log_predictions` without HTTP, including failure handling |
| `tests/system/test_router.py` | `/health` stays unprefixed, `/model/info`, OpenAPI |
| `tests/model_registry/test_service.py` | **The BYTEA round-trip rebuilds a working predictor**; one active version |
| `tests/training_data/test_service.py` | JSONB payload building: NaN → null, numpy → native |
| `tests/test_layering.py` | **The dependency arrows in §1 actually hold** |
| `tests/test_error_handling.py` | An unreachable database → 503, not 500 |
| `tests/test_config.py` | `DATABASE_URL` normalisation; derived paths; version metadata; **the seven settings have no code-side default** |

Each test directory needs an `__init__.py`: `tests/__init__.py` exists, and several modules
share a basename (`test_router.py`, `test_service.py`), which pytest would otherwise reject
as duplicates.

**No test touches PostgreSQL.** The session layer is faked and the model is built from
joblib bytes in memory, so `uv run pytest` passes with the database down or `DATABASE_URL`
unset — worth verifying occasionally, since it is easy to lose by accident. The cost is
that the schema, the migrations and the JSONB/BYTEA round-trip are **not** covered by the
suite; §8 covers them against the real database instead.

**Expect** `93 passed` in about six seconds.

The suite trains a real model, so assertions are about real behaviour. The previous suite
asserted `status_code in [200, 503]`, which passed whether the service worked or not; those
are gone.

**Confirm the tests can actually fail** — a test suite you have never seen fail is not
evidence of anything:

Reintroduce the original bug by deleting the `normalise` step from
`ml/schema.py:build_preprocessor` — the step that maps a served `None` onto the `NaN` the
imputer recognises:

```bash
cp src/dropout_service/ml/schema.py /tmp/pipeline.bak
# delete the ("normalise", FunctionTransformer(...)) tuple from build_preprocessor
$EDITOR src/dropout_service/ml/schema.py

uv run pytest tests/ml/test_no_train_serve_skew.py -q
# FAILED test_omitted_category_matches_the_value_used_at_training_time
# FAILED test_an_unseen_category_would_have_produced_a_different_answer

cp /tmp/pipeline.bak src/dropout_service/ml/schema.py
uv run pytest -q          # 93 passed again
```

The layering tests can be checked the same way — add
`from dropout_service.ml.training import data` to `ml/predict.py` and
`tests/test_layering.py` fails, naming the leaked module.

Note that changing `CATEGORICAL_FILL` in `ml/schema.py` does *not* break the primary guard, and
should not: that constant is read by the shared pipeline, so training and serving move
together and stay consistent. Only a change that makes the two paths disagree — like dropping
the `normalise` step — is a real regression. That is the distinction the test encodes.

---

## 8. The database

Required. It holds the training data, the trained model and the audit log, so every step
after `uv sync` depends on it.

**Commands**

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/dropout_db"
uv run alembic upgrade head
```

**File sequence**

| # | File : function | What it does |
|---|---|---|
| 1 | `alembic.ini` | Points at `alembic/`, prepends `src` to `sys.path` |
| 2 | `alembic/env.py:run_migrations_online` | Entry point |
| 3 | `config.py:get_settings` | Read `DATABASE_URL` (no credentials in any file) |
| 4 | `alembic/env.py` | Imports `database.py:Base` **and every slice's `models`** |
| 5 | `alembic/env.py` | `CREATE SCHEMA IF NOT EXISTS ml` |
| 6 | `alembic/versions/0001_...py:upgrade` | Create `ml.predictions` + the `created_at` index |
| 7 | `alembic/versions/0002_full_schema.py:upgrade` | Create `users`, `students`, `training_records`, `model_versions`, `analysis_reports`; add the FKs to `predictions` |

Step 5 exists because Alembic stores its own `alembic_version` table inside the `ml` schema
and creates it *before* running any migration — so the schema must already exist or the first
upgrade fails with `InvalidSchemaName`.

**Step 4 is load-bearing, and silently destructive if you get it wrong.** Tables live in
their own slice (`predictions/models.py`), not in `database.py`, so they are only registered
on `Base.metadata` once that module has been imported. `env.py` imports it explicitly:

```python
from dropout_service.model_registry import models as _model_registry_models  # noqa: E402,F401
from dropout_service.predictions import models as _predictions_models  # noqa: E402,F401
from dropout_service.students import models as _students_models  # noqa: E402,F401
from dropout_service.training_data import models as _training_data_models  # noqa: E402,F401
from dropout_service.users import models as _users_models  # noqa: E402,F401
```

Delete that line and `alembic revision --autogenerate` sees the table in the database but
not in the metadata, and concludes you meant to remove it:

```python
def upgrade() -> None:
    op.drop_index(op.f("ix_ml_predictions_created_at"), table_name="predictions", schema="ml")
    op.drop_table("predictions", schema="ml")
```

That is the real output of running it without the import. Applying such a migration would
destroy the audit log. **Add a line there for every new slice that owns tables**, and after
adding one, confirm an autogenerate against an up-to-date database produces an empty
migration.

**Verify**

```bash
psql "$DATABASE_URL" -c "\dt ml.*"              # six tables
psql "$DATABASE_URL" -c "\d ml.model_versions"

# The artefact really is bytes, and exactly one version is active
psql "$DATABASE_URL" -tAc "select length(artifact), algorithm, is_active
                           from ml.model_versions;"
psql "$DATABASE_URL" -tAc "select count(*) from ml.model_versions where is_active;"  # 1

# Make a prediction, then confirm it was logged and linked
curl -s -o /dev/null -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' -d @/tmp/student.json
sleep 1
psql "$DATABASE_URL" -c "select id, student_id, model_version_id, risk_level
                         from ml.predictions order by id desc limit 5;"
curl -s 'localhost:8000/predictions?page=1&page_size=5' | python3 -m json.tool
```

Confirm the metadata registration is intact — the check that protects the audit log:

```bash
# Against a database already at head, this must generate an EMPTY migration.
uv run alembic revision --autogenerate -m "should-be-empty"
sed -n '/^def upgrade/,/^def downgrade/p' alembic/versions/*should_be_empty*
# Expect only:  pass
# If it contains drop_table, a models import is missing from alembic/env.py.
rm alembic/versions/*should_be_empty*
```

Two constraints are worth checking by hand, because the test suite cannot:

```bash
# Exactly one active model, enforced by a partial unique index rather than by hope.
psql "$DATABASE_URL" -c "update ml.model_versions set is_active = true;"
# ERROR: duplicate key value violates unique constraint "uq_model_versions_single_active"

# The migration reverses cleanly.
uv run alembic downgrade base && uv run alembic upgrade head
```

A bare `postgresql://` URL is rewritten to `postgresql+psycopg://` by
`config.py:Settings._use_psycopg3_driver`, because SQLAlchemy maps the bare scheme to
psycopg2 — which this project does not install. Hosted providers emit exactly that bare
form, so without the rewrite prediction logging would silently never work in production.

---

## 9. Deploy

**Commands**

```bash
docker build -t dropout-service:1.0.0 .
docker run -p 8000:8000 --env-file .env dropout-service:1.0.0
```

**File sequence**

1. `.dockerignore` — excludes `.venv`, `data/`, `tests/`, `docs/`, caches. `README.md` is
   explicitly kept: `pyproject.toml` declares it as the project readme, so the wheel will
   not build without it
2. `Dockerfile` build stage — `uv sync --frozen --no-dev` against `uv.lock`
3. `Dockerfile` runtime stage — copies `.venv`, `src`, `alembic`; creates `appuser`. No
   model and no training data: the model is read from the database at startup
4. `CMD` — `uvicorn dropout_service.main:app --port ${PORT:-8000}`
5. `HEALTHCHECK` in the Dockerfile — the only health probe; there is no
   platform-specific config file

The image ships neither `models/` nor `data/`: the active model version comes from the
database at startup, so one image works against any environment and nothing needs
rebuilding to ship a retrained model. Remember from §4 that the artefact itself must be
retrained whenever Python modules move.

**Verify**

```bash
docker run -d -p 8000:8000 --env-file .env --name dropout-test dropout-service:1.0.0
sleep 5
curl -s localhost:8000/health
docker exec dropout-test ls /app/data /app/models 2>&1   # neither must exist
docker stop dropout-test && docker rm dropout-test

# And confirm the config really is required: no --env-file means no start.
docker run --rm dropout-service:1.0.0 2>&1 | tail -3   # ValidationError
```

Push the image to any registry and run it wherever you like — there is no
platform-specific configuration:

```bash
docker tag dropout-service:1.0.0 <registry>/dropout-service:1.0.0
docker push <registry>/dropout-service:1.0.0
```

Two things the host must do: supply the required settings at run time (`--env-file`, or the
platform's own secret mechanism — never baked into the image, which would recreate the
defaults the config removed), and run `alembic upgrade head` before the new version serves
traffic if you use the database. The container's own `HEALTHCHECK` already probes
`/health`.

---

## 10. Authentication (planned)

**Not implemented.** Every endpoint is currently open: anyone who can reach the service can
request predictions and page through the entire prediction history, which contains personal
student data. This is the largest outstanding gap and the next piece of work.

Because the `web-client` talks to this service directly from the browser, a shared API key
would not be a secret — it would ship in the JavaScript bundle. The login page implies real
user accounts, so the intended design is:

The structure is now shaped for this: auth is **one new directory**, not an edit to four
shared files.

```
src/dropout_service/auth/
  router.py      POST /auth/login, GET /auth/me
  schemas.py     LoginRequest, TokenResponse, UserRead
  service.py     authenticate(), get_current_user()
  security.py    JWT encode/decode  (bcrypt hashing already in scripts/create_user.py)

# the table itself already exists: users/models.py + migration 0002
```

| Piece | Approach |
|---|---|
| Storage | **Already done**: `ml.users` exists (`users/models.py`), and `scripts/create_user.py` creates accounts with bcrypt |
| Login | `POST /auth/login` → short-lived JWT; `GET /auth/me` |
| Hashing | `bcrypt` (via `passlib`), never a bare SHA |
| Verification | `get_current_user` in `auth/service.py`, exposed as `CurrentUserDep` in `dependencies.py` |
| Protected | `/predict`, `/predict/batch`, `/predictions` |
| Open | `/health` and `/` only — the `system` router |

Three wiring points, each already marked with a comment in the code:

1. `main.py` — `app.include_router(auth_router)`
2. `dependencies.py` — add `CurrentUserDep`, mirroring the existing `Annotated` aliases
3. `alembic/env.py` — nothing to add; `users/models.py` is already imported there

Put the blanket requirement on the `include_router` call for the protected routers, not on
`system_router` — that one carries `/health`, which the container's `HEALTHCHECK` needs to
reach unauthenticated.

Worth deciding at the same time: who may read `/predictions`. A per-student risk history is
sensitive, and advisers, administrators and researchers do not need the same access — hence
`role` in the table above.

Until this lands, keep the service on a private network and `CORS_ORIGINS` restricted to
known frontend origins.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/health` → `"status":"degraded"` | No active model version, or the database is down | `uv run scripts/train.py`; check `/health`'s `database` field |
| `/predict` → 503 | Same | As above; check startup logs |
| `autogenerate` emits `drop_table` | A slice's `models` import is missing from `alembic/env.py` | Add it; see §8 |
| `/predictions` → 503 | Database unreachable | Check the logs and `/health`; the connection string is in `.env` |
| `/health` → `degraded`, `model_loaded: false` | No active `model_versions` row | `uv run scripts/train.py` |
| `RuntimeError: No training records` | Database not seeded | `uv run scripts/seed_db.py` |
| `relation "ml.training_records" does not exist` | Migrations not applied | `uv run alembic upgrade head` |
| `ModuleNotFoundError: psycopg2` | Pre-normalisation URL | Use `postgresql://...`; it is rewritten automatically |
| `InvalidSchemaName: schema "ml" does not exist` | Table created outside Alembic | `uv run alembic upgrade head` |
| 422 on a seemingly valid body | Closed vocabulary, or an unknown field | Read `detail`; check `/docs` |
| 422 on `area_materia_critica` | Sending `"N/A"` or `"Unknown"` | Omit the field or send `null` |
| `ModuleNotFoundError: dropout_service` | Package not installed | `uv sync` (installs the project itself) |
| `ModuleNotFoundError` naming a module **when loading the model** | The stored artefact predates a module move; the pickle holds import paths | `uv run scripts/train.py` to rebuild it |
| `ValidationError` naming settings at startup | No `.env`, or a key missing from it | `cp .env.example .env`; in Docker pass `--env-file .env` |
| 500 from a database endpoint | Should not happen — `SQLAlchemyError` is mapped to 503 | Check the logs; if it is genuinely a 500, the handler in `main.py` was bypassed |
| CORS error in the browser | Origin not allowed | Add it to `CORS_ORIGINS` |
| Predictions all look similar | Model is weak by nature (AUC 0.77) | Expected; use ranking, not absolutes |

Read the logs first — startup reports whether the model loaded and whether the database was
reachable, which explains most of the above.

---

## 12. Design decisions, and why

Six rules shape this codebase. Each exists because of a specific bug or a specific
failure it would otherwise allow.

### Every feature is declared exactly once

All 34 features are fields on `StudentFeatures` in `ml/schema.py`; `CATEGORICAL_FEATURES`
and `NUMERIC_FEATURES` are *derived* from it by introspection.

The previous version hand-wrote that list in five files — the ORM model, a DB loader, a seed
script, two Pydantic schemas and an endpoint. Two copies drifted: training filled missing
categories with `"Unknown"` while the API defaulted to `"N/A"`. Because the encoder used
`handle_unknown="ignore"`, the mismatch produced an all-zero row instead of an error, and
silently changed the prediction for the ~50% of students with no critical subject area
(0.2449 vs 0.2081 on the documented example). Nothing failed; the numbers were just wrong.

Deriving the lists makes that class of bug unrepresentable, and
`tests/ml/test_schema.py::test_feature_lists_cover_every_field_exactly_once` fails if a
field escapes.

### Anything that learns from data lives inside the pipeline

Imputation, encoding and scaling are all steps built in `ml/schema.py`, never in
`ml/training/data.py`.

They live in `schema.py` rather than a module of their own because the two halves belong
together — the feature declaration says *what* the model consumes, the pipeline says *how*
each one is transformed, and they have to agree. There is also a hard constraint on where
the preprocessing may live at all: the pickled model references `coerce_missing_to_nan` by
import path and the serving path unpickles it, so the function must sit in a module the API
can import by itself. Putting it under `ml/training/` would make loading a model drag the
training code into the API and break the rule below.

Two reasons. First, **no leakage**: the old `clean_data` computed medians over the whole
dataset before splitting, so every cross-validation fold saw statistics drawn partly from
its own validation rows, and the reported score was optimistic. Inside a `Pipeline`,
scikit-learn refits each step per fold.

Second, **one code path**: training and serving reach the exact same fitted transformers, so
they cannot disagree about anything.

That second property needed care. `SimpleImputer` only treats `np.nan` as missing, but the
two entry points spell "absent" differently — `pd.read_csv` yields `NaN`, a Pydantic optional
field yields `None`. A served `None` therefore slipped past the imputer and reached the
encoder as an unseen category: the original bug, reborn. Hence
`ml/schema.py:coerce_missing_to_nan`, which normalises both to `NaN` inside the shared
pipeline rather than patching it at the serving edge.

### The threshold is calibrated, and calibrated on the right data

`pipeline.predict()` hardcodes 0.5. For an early-warning system that is the wrong operating
point: missing a student who drops out costs a degree, while a false alarm costs one advising
conversation. `ml/training/train.py:select_threshold` picks the threshold meeting
`TARGET_RECALL` (0.80),
stores it alongside the artefact in `ml.model_versions`, and `Predictor` compares
`predict_proba` against it. Recall goes from 0.58 to 0.79.

It is selected from **out-of-fold predictions on the training set**, not the test set.
Choosing it on the test set would tune the operating point to the rows used to report
performance — a subtler form of the same leakage as above.

### Grouped by feature, not by layer

Each slice (`predictions/`, `system/`, later `auth/`) owns its router, schemas, tables and
logic. The alternative — one `routes.py`, one `schemas.py`, one `models.py` for the whole
service — means every new capability edits the same four files that every other capability
also touches. Adding auth here is adding a directory.

`ml/` is the exception and deliberately so: it has no HTTP surface, so it is a domain
library that the slices consume, not a peer of them.

The dependency arrows are not a convention, they are tested
(`tests/test_layering.py`), because two of them have real production consequences:

- **Serving must never import `ml/training/`.** Those modules read the CSV from `data/`,
  which is excluded from the Docker image. An accidental import would pass every local test
  and fail only once deployed.
- **`ml/` must never import `fastapi` or the slices.** That keeps the model usable from a
  notebook, a batch job, or a future scheduled retraining task — none of which should need
  a web framework installed.

A related consequence worth knowing: the stored artefact pickles
`ml/schema.py:coerce_missing_to_nan` by import path, so moving Python modules invalidates
every previously trained artefact. That is why this restructure retrained the model, and
why a production release that moves modules must ship a fresh one.

**Why a root package rather than `src/ml/` directly.** The slices could sit at the top of
`src/` with no wrapper — `src/ml/`, `src/predictions/`, `src/config.py`, `src/main.py` —
which is the layout the widely-read `fastapi-best-practices` guide uses. Shorter imports,
one less directory level. Two things argue against it *here*.

First, the collision surface. Flat layout makes every one of those names a **global**
top-level import: `ml`, `config`, `main`, `database`, `dependencies`, `predictions`,
`router`, `system`. Six of those eight are already real packages on PyPI (`main` and
`system` are the exceptions). Nothing clashes in this project's current environment, but
a flat layout installs those names straight into `site-packages`, so the day a dependency
pulls in a top-level `config` or `ml`, which one wins depends on `sys.path` ordering.

Second — and this is the less obvious one — **loose top-level modules are silently dropped
from the wheel.** A scratch build declaring `packages = ["src/ml", "src/predictions"]` with
a `src/config.py` beside them produced a wheel containing `ml/` and `predictions/` and no
`config.py` at all. No error, no warning. Every loose module would need its own explicit
entry, and omitting one breaks only after deployment.

The guide's layout is not wrong, it answers a different question: that project is
Docker-only, never pip-installed, and runs with `src` as the working directory, so none of
the above applies. This project *is* an installable distribution — `uv sync` installs it —
and that is precisely why `scripts/train.py` and `scripts/analyze.py` contain no
`sys.path` manipulation and the `Dockerfile` needs no `PYTHONPATH` for imports to resolve.
Going flat would mean reinstating those hacks or accepting the collision surface.

The cost is real and worth stating plainly: every import carries the 16 characters of
`dropout_service.` — `from dropout_service.ml.predict import Predictor` against
`from ml.predict import Predictor`. That is the whole price. The same reasoning, from a
different angle, rules out a sibling `src/api/` package — see the end of *KISS first*
below.

### State lives in the database; `ml/` holds none of it

The trained artefact is a `BYTEA` column and its metrics a `JSONB` one. The split is not
arbitrary: a fitted scikit-learn pipeline is **not** JSON-serialisable — `json.dumps`
raises, because it carries numpy arrays, learned one-hot categories, scaler statistics and
a reference to `coerce_missing_to_nan`. A JSON-only store would need a bespoke serialiser
that works only for `LogisticRegression`, and `train.py` picks the winner among five
algorithms, so it would break silently the day RandomForest or the calibrated SVM won.
The metrics genuinely *are* JSON, so they get a queryable column.

Training features are stored as **one `JSONB` payload, not 34 columns**, for exactly the
reason the first rule exists: a mirrored column list would be a second copy of the feature
set, free to drift from `StudentFeatures` — and it would need a migration for every new
feature. Individual values stay queryable via `features->>'nota_promedio'`.

`ml/` stays free of SQLAlchemy as well as FastAPI. `Predictor.from_bytes` takes bytes and a
dict; `model_registry/service.py` is what reads the row. That is what keeps the model usable
from a notebook, a batch job or a future scheduled retrain, none of which should need a web
framework or a database session to import. `tests/test_layering.py` enforces it.

Two guarantees are pushed down into the schema rather than left to application code: a
partial unique index on `is_active` makes "exactly one active model" impossible to violate,
and foreign keys from `predictions` to `model_versions` and `students` mean an audit row
stays interpretable after a retrain.

### KISS first; SOLID where it pays

Applied: one source of truth; dependency injection via `Depends` instead of a global `_state`
dict; a `create_app()` factory so tests can override settings; `GridSearchCV` driven by a
`PARAM_GRIDS` dict instead of 60 lines of `if/elif` (adding an algorithm is now a dict entry,
not an edit to a function); thin routers with the logic in `service.py`, so it can be tested
without HTTP.

Deliberately *not* applied: no `Protocol` interfaces, no repository abstraction, no DI
container, and no `exceptions.py` holding nothing. There is one model format, one data source
and one database. An interface with a single implementation is indirection without benefit;
the time to introduce one is when a second implementation actually arrives.

The same restraint decided against a sibling `src/api/` package: that would make `api` a
*global* top-level import name, ship two unrelated top-level packages from one wheel, and
imply that HTTP is a peer of the domain rather than a way of delivering it. That is the
same trade-off as *Why a root package rather than `src/ml/` directly* above, reached from
the other direction.

`training_records` has a `source` column so verified outcomes could be accumulated
alongside the seeded CSV rows later, but nothing writes them yet — the column exists
because adding it afterwards would mean a migration over real data, not because the feature
is half-built.
