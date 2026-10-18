"""Database engine and SQLAlchemy base."""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


def make_engine(database_url: str):
    kwargs = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(database_url, **kwargs)
    if database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(connection, _record):
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return engine


def make_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
