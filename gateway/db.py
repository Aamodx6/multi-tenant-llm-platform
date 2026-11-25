from __future__ import annotations


def normalize_database_url(database_url: str) -> str:
    """Normalize PostgreSQL URLs to the SQLAlchemy 2.0 psycopg v3 driver dialect."""
    if database_url.startswith("postgres://"):
        return "postgresql+psycopg://" + database_url[len("postgres://") :]
    if database_url.startswith("postgresql://"):
        return "postgresql+psycopg://" + database_url[len("postgresql://") :]
    return database_url
