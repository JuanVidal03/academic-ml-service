from typing import Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field

Genero = Literal["F", "M", "Otro"]
Modalidad = Literal["Diurna", "Nocturna"]
TipoAdmision = Literal["Regular", "Transferencia", "Reingreso", "Homologacion"]
NivelEducativoPadres = Literal[
    "Ninguno", "Primaria", "Secundaria", "Tecnico", "Universitario", "Posgrado"
]
AreaMateriaCritica = Literal[
    "Matematicas y Ciencias Basicas",
    "Fundamentos Tecnicos",
    "Humanidades",
    "Economicas",
    "Salud",
]
SiNo = Literal["Si", "No"]


class StudentFeatures(BaseModel):
    """A student's features at the moment of prediction.

    Field order is irrelevant to the model; fields are grouped for readability only.
    """

    model_config = ConfigDict(
        extra="forbid",  # a misspelled field is an error, not a silently ignored default
        json_schema_extra={
            "example": {
                "semestre": 3,
                "programa": "Ingenieria de Sistemas",
                "materias_matriculadas": 6,
                "materias_aprobadas": 4,
                "materias_reprobadas": 2,
                "materias_repetidas": 1,
                "nota_promedio": 3.2,
                "promedio_semestre_anterior": 3.0,
                "promedio_acumulado": 3.1,
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
                "uso_tutorias": "Si",
            }
        },
    )

    # --- Academic load and performance ---
    semestre: int = Field(..., ge=1, le=20, description="Current semester")
    programa: str = Field(..., min_length=1, description="Academic program")
    materias_matriculadas: int = Field(..., ge=0, le=12, description="Subjects enrolled")
    materias_aprobadas: int = Field(..., ge=0, le=12, description="Subjects passed")
    materias_reprobadas: int = Field(..., ge=0, le=12, description="Subjects failed")
    materias_repetidas: int = Field(0, ge=0, le=12, description="Subjects repeated")
    materias_canceladas: int = Field(0, ge=0, le=12, description="Subjects cancelled")
    semestres_repetidos: int = Field(0, ge=0, le=10, description="Semesters repeated")
    numero_materias_con_perdida_historica: int = Field(
        0, ge=0, le=20, description="Subjects ever failed"
    )
    nota_promedio: float = Field(..., ge=0, le=5, description="Current term GPA (0-5)")
    promedio_semestre_anterior: float = Field(..., ge=0, le=5, description="Previous term GPA")
    promedio_acumulado: float = Field(..., ge=0, le=5, description="Cumulative GPA")
    area_materia_critica: AreaMateriaCritica | None = Field(
        None,
        description=(
            "Subject area the student struggles with most. Omit or send null when the "
            "student has no critical area (true for ~50% of students); the preprocessing "
            "pipeline supplies the fill value used at training time."
        ),
    )

    # --- Attendance and study habits ---
    porcentaje_asistencia: float = Field(..., ge=0, le=100, description="Attendance percentage")
    inasistencias: int = Field(0, ge=0, description="Count of absences")
    horas_estudio_semana: float = Field(..., ge=0, le=120, description="Weekly study hours")

    # --- Demographics ---
    edad: int = Field(..., ge=15, le=80, description="Age in years")
    genero: Genero = Field(..., description="Gender")
    estrato: int = Field(..., ge=1, le=6, description="Colombian socioeconomic stratum (1-6)")

    # --- Admission and logistics ---
    tipo_admision: TipoAdmision = Field("Regular", description="Admission route")
    modalidad: Modalidad = Field("Diurna", description="Day or evening programme")
    distancia_a_universidad_km: float = Field(0, ge=0, description="Home-to-campus distance (km)")

    # --- Employment ---
    trabaja: SiNo = Field("No", description="Currently employed")
    horas_trabajo_semana: int = Field(0, ge=0, le=100, description="Weekly work hours")

    # --- Economic situation ---
    ingreso_familiar: float = Field(..., ge=0, description="Monthly household income (COP)")
    recibe_beca: SiNo = Field("No", description="Holds a scholarship")
    beca_perdida_en_periodo: SiNo = Field("No", description="Lost a scholarship this term")
    dificultad_economica: SiNo = Field("No", description="Reports financial hardship")

    # --- Family and support ---
    tiene_hijos: SiNo = Field("No", description="Has children")
    nivel_educativo_padres: NivelEducativoPadres = Field(
        ..., description="Highest parental education level"
    )
    apoyo_familiar_percibido: int = Field(..., ge=1, le=5, description="Perceived family support")

    # --- Engagement ---
    satisfaccion_con_programa: int = Field(..., ge=1, le=5, description="Programme satisfaction")
    participacion_extracurricular: SiNo = Field("No", description="Takes part in activities")
    uso_tutorias: SiNo = Field("No", description="Uses tutoring services")


def _is_categorical(annotation: object) -> bool:
    """Decide whether a field annotation denotes a categorical feature.

    Handles ``str``, ``Literal[...]``, and the optional forms of both (``X | None``).
    """
    if get_origin(annotation) is Literal:
        return True
    args = [a for a in get_args(annotation) if a is not type(None)]
    if args:
        # An optional or union type: classify by its single meaningful member.
        return _is_categorical(args[0])
    return annotation is str


CATEGORICAL_FEATURES: list[str] = [
    name
    for name, field in StudentFeatures.model_fields.items()
    if _is_categorical(field.annotation)
]

NUMERIC_FEATURES: list[str] = [
    name
    for name, field in StudentFeatures.model_fields.items()
    if not _is_categorical(field.annotation)
]

#: Column order the pipeline expects. Categorical first, then numeric.
ALL_FEATURES: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

#: Target column in the source CSV, and the mapping to a binary label.
TARGET_COLUMN = "deserto"
TARGET_MAP = {"Si": 1, "No": 0}

#: Fill value for a missing categorical. Used by the pipeline at *both* training and
#: serving time, so the two can never disagree.
CATEGORICAL_FILL = "Unknown"
