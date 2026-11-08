#!/usr/bin/env python3
"""Create a user account.

Usage:
    uv run scripts/create_user.py --email someone@universidad.edu.co --role admin
"""

import argparse
import getpass
import sys

from dropout_service.auth.security import hash_password
from dropout_service.config import configure_logging, get_settings
from dropout_service.database import create_db_engine, create_session_factory
from dropout_service.users.models import User
from dropout_service.users.service import get_by_email

MIN_PASSWORD_LENGTH = 12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True)
    parser.add_argument("--role", default="adviser", choices=["adviser", "admin"])
    args = parser.parse_args()

    # Stored lowercase because lookups normalise the same way; without this a mixed-case
    # address could be created and then never match at login.
    email = args.email.strip().lower()

    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Confirm: "):
        sys.exit("Passwords do not match.")
    if len(password) < MIN_PASSWORD_LENGTH:
        sys.exit(f"Use at least {MIN_PASSWORD_LENGTH} characters.")

    configure_logging()
    engine = create_db_engine(get_settings().database_url)
    with create_session_factory(engine)() as session:
        if get_by_email(session, email) is not None:
            sys.exit(f"{email} already exists.")
        session.add(User(email=email, password_hash=hash_password(password), role=args.role))
        session.commit()

    print(f"Created {args.role} {email}")


if __name__ == "__main__":
    main()
