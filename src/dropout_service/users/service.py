"""User lookups."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from dropout_service.users.models import User


def get_by_email(session: Session, email: str) -> User | None:
    """Find a user by email, case-insensitively.

    Email is case-insensitive in practice, so a login must not fail because someone typed
    a capital letter. The column has a unique index on the stored value; normalising on
    write keeps that meaningful.
    """
    return session.scalars(select(User).where(User.email == email.strip().lower())).first()


def get_by_id(session: Session, user_id: int) -> User | None:
    return session.get(User, user_id)
