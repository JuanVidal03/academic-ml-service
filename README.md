# Dropout Prediction ML Service

Machine learning service for predicting academic dropout risk. Built with FastAPI, scikit-learn, and SQLAlchemy.

## Features

- **Model Training**: Compares multiple ML algorithms (LogisticRegression, RandomForest, HistGradientBoosting, GradientBoosting, SVM) using cross-validation
- **REST API**: FastAPI endpoints for predictions, model info, and health checks
- **Database Integration**: PostgreSQL with SQLAlchemy for storing training data and prediction logs
- **Production Ready**: Dockerfile and Railway configuration included

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- PostgreSQL (optional - service works with CSV fallback)

### Installation

```bash
# Clone and navigate to the project
cd ml-service

# Install dependencies
uv sync

# Copy environment file
cp .env.example .env
# Edit .env with your database URL if using PostgreSQL
```

### Run Analysis

Generate a data analysis report:

```bash
uv run scripts/analyze.py
```

This creates `reports/analysis.md` and `reports/analysis.json` with:
- Dataset statistics
- Feature correlations with dropout
- Mutual information scores
- Random Forest feature importance
- Dropout rates by category

### Train the Model

```bash
uv run scripts/train.py
```

This will:
1. Load data from database (or CSV fallback)
2. Compare 5 ML algorithms using 5-fold cross-validation
3. Tune hyperparameters for the best model
4. Train final model and evaluate on test set
5. Save model to `models/model.joblib` and metrics to `models/metrics.json`

### Run the API

```bash
# Development
PYTHONPATH=src uv run uvicorn dropout_ml.api.main:app --reload --port 8000

# Production
PYTHONPATH=src uv run uvicorn dropout_ml.api.main:app --host 0.0.0.0 --port 8000
```

### Seed Database (Optional)

If using PostgreSQL, seed the training data:

```bash
# Set DATABASE_URL in .env first
uv run scripts/seed_db.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (for Railway/load balancers) |
| GET | `/model/info` | Model metadata and test metrics |
| POST | `/predict` | Predict dropout risk for one student |
| POST | `/predict/batch` | Predict for multiple students (max 100) |
| GET | `/predictions` | List prediction history (paginated) |
| POST | `/records` | Add verified training record |

### Example: Predict Dropout Risk

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
    "nota_promedio": 3.2,
    "promedio_semestre_anterior": 3.0,
    "promedio_acumulado": 3.1,
    "creditos_perdidos": 6,
    "materias_canceladas": 0,
    "semestres_repetidos": 0,
    "numero_materias_con_perdida_historica": 2,
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

Response:
```json
{
  "deserta": false,
  "probabilidad": 0.35,
  "riesgo": "medio"
}
```

## Development

```bash
# Run tests
uv run pytest -v

# Lint code
uv run ruff check .

# Format code
uv run ruff format .
```

## License

MIT
