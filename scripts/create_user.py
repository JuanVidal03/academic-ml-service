#!/usr/bin/env python3
"""Create a user account.

The users table exists for the authentication work that comes next; this makes it usable
in the meantime.

Usage:
    uv run scripts/create_user.py --email someone@example.com --role admin
"""

import argparse
import getpass
import sys

import bcrypt

from dropout_service.config import configure_logging, get_settings
from dropout_service.database import create_db_engine, create_session_factory
from dropout_service.users.models import User


def hash_password(plaintext: str) -> str:
    """Hash with bcrypt, which is deliberately slow; never a bare SHA digest."""
    return bcrypt.hashpw(plaintext.encode(), bcrypt.gensalt()).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True)
    parser.add_argument("--role", default="adviser", choices=["adviser", "admin"])
    args = parser.parse_args()

    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Confirm: "):
        sys.exit("Passwords do not match.")
    if len(password) < 12:
        sys.exit("Use at least 12 characters.")

    configure_logging()
    engine = create_db_engine(get_settings().database_url)
    with create_session_factory(engine)() as session:
        session.add(User(email=args.email, password_hash=hash_password(password), role=args.role))
        session.commit()

    print(f"Created {args.role} {args.email}")


if __name__ == "__main__":
    main()
