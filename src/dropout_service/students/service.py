"""Student lookups."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from dropout_service.students.models import Student


def get_by_codigo(session: Session, codigo: str) -> Student | None:
    return session.scalars(select(Student).where(Student.codigo == codigo)).first()


def upsert_by_codigo(session: Session, codigo: str, programa: str | None = None) -> Student:
    """Return the student with this code, creating the row if it is new.

    Flushes rather than commits, so the caller controls the transaction boundary.
    """
    student = get_by_codigo(session, codigo)
    if student is None:
        student = Student(codigo=codigo, programa=programa)
        session.add(student)
        session.flush()
    elif programa and student.programa != programa:
        student.programa = programa
    return student
