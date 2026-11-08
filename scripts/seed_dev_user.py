#!/usr/bin/env python3
"""Create the local development account.

Separate from create_user.py on purpose. That script enforces a 12-character minimum; this
one deliberately creates a 7-character password for convenience while developing, so it is
gated behind an explicit flag that nobody types by accident. The weak credential therefore
cannot reach production through the normal path.

Usage:
    uv run scripts/seed_dev_user.py --i-know-this-is-insecure
"""

import argparse
import sys

from dropout_service.auth.security import hash_password
from dropout_service.config import configure_logging, get_settings
from dropout_service.database import create_db_engine, create_session_factory
from dropout_service.users.models import User
from dropout_service.users.service import get_by_email

DEV_EMAIL = "test@universidad.edu.co"
DEV_PASSWORD = "test123"  # noqa: S105 - a development fixture, gated behind the flag below
DEV_ROLE = "admin"  # admin so the account can exercise the admin-only endpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--i-know-this-is-insecure",
        action="store_true",
        dest="confirmed",
        help="Required. Confirms you understand this creates a weak, well-known password.",
    )
    args = parser.parse_args()

    if not args.confirmed:
        sys.exit(
            "Refusing to run.\n\n"
            f"This creates {DEV_EMAIL} with the password '{DEV_PASSWORD}', which is public\n"
            "knowledge and far too short. It exists only for local development.\n\n"
            "If this is a development database, re-run with --i-know-this-is-insecure.\n"
            "For a real account use: uv run scripts/create_user.py --email ... --role ..."
        )

    configure_logging()
    engine = create_db_engine(get_settings().database_url)
    with create_session_factory(engine)() as session:
        existing = get_by_email(session, DEV_EMAIL)
        if existing is not None:
            print(f"{DEV_EMAIL} already exists (id={existing.id}, role={existing.role})")
            return
        user = User(email=DEV_EMAIL, password_hash=hash_password(DEV_PASSWORD), role=DEV_ROLE)
        session.add(user)
        session.commit()
        user_id = user.id

    print("\n" + "=" * 62)
    print("Development account created")
    print("=" * 62)
    print(f"  id       : {user_id}")
    print(f"  email    : {DEV_EMAIL}")
    print(f"  password : {DEV_PASSWORD}")
    print(f"  role     : {DEV_ROLE}")
    print("\n  Do not expose this service publicly while this account exists.")


if __name__ == "__main__":
    main()
