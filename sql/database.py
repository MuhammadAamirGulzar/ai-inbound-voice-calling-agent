import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

load_dotenv()

SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://ai_caller_user:PASSWORD@localhost:5432/ai_caller"
)

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def ensure_columns():
    """
    Minimal additive migration: create_all() only creates missing tables,
    it never adds columns to existing ones. This adds the columns the
    streaming voice engine needs to databases created before it existed.
    Safe to run on every startup (no-op when columns already exist).
    """
    from sqlalchemy import inspect, text

    json_type = "JSONB" if engine.dialect.name == "postgresql" else "JSON"
    wanted = {
        "agent_configurations": [("voice_settings", json_type)],
        "call_logs": [("metrics", json_type)],
    }
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, columns in wanted.items():
            if table not in inspector.get_table_names():
                continue
            existing = {c["name"] for c in inspector.get_columns(table)}
            for name, col_type in columns:
                if name not in existing:
                    conn.execute(text(
                        f"ALTER TABLE {table} ADD COLUMN {name} {col_type}"))
                    print(f"[db] added column {table}.{name}")
