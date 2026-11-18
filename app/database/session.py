from contextlib import asynccontextmanager
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from app.config import settings
from app.database.models import Base

import asyncio
db_write_lock = asyncio.Lock()

engine = create_async_engine(
    settings.database_url.replace("sqlite:///", "sqlite+aiosqlite:///"),
    echo=settings.debug_mode,
    poolclass=NullPool,
    # SQLite is written to concurrently by message handlers and background
    # loops: a generous busy timeout avoids spurious "database is locked".
    connect_args={"timeout": 30},
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Columns added after the original schema. SQLite's CREATE TABLE does not
# alter existing tables, so init_db applies any missing columns via
# ALTER TABLE before create_all runs.
_COLUMN_MIGRATIONS = {
    "emotional_state": {
        "confidence": "REAL DEFAULT 0.5",
        "embarrassment": "REAL DEFAULT 0.0",
        "competitiveness": "REAL DEFAULT 0.5",
        "boredom": "REAL DEFAULT 0.2",
    },
    "relationships": {
        "comfort": "REAL DEFAULT 0.0",
        "jealousy": "REAL DEFAULT 0.0",
        "relationship_type": "VARCHAR(30) DEFAULT 'friend'",
        "is_canonical": "BOOLEAN DEFAULT 0",
    },
    "recent_phrases": {
        "target_user_id": "INTEGER",
        "topic": "VARCHAR(300)",
    },
    "conversation_states": {
        "initiations_without_engagement": "INTEGER DEFAULT 0",
        "last_initiation_at": "DATETIME",
        "recent_topics": "JSON",
        "pending_topic": "VARCHAR(500)",
        "typing_register": "VARCHAR(20) DEFAULT 'normal'",
        "typing_mode_until": "DATETIME",
    },
    "conversations": {
        "last_activity_at": "DATETIME",
    },
    "memories": {
        "reinforcement": "INTEGER DEFAULT 0",
    },
    "messages": {
        "reply_to_user_id": "INTEGER",
    },
    "self_model": {
        "stances": "JSON",
        "style_alarms": "JSON",
    },
    "relationships": {
        "irritation_streak": "REAL DEFAULT 0.0",
        "last_message_at": "DATETIME",
    },
}


async def _run_column_migrations(conn) -> None:
    for table, columns in _COLUMN_MIGRATIONS.items():
        # Check the table exists first (fresh databases are handled by create_all)
        result = await conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"),
            {"t": table},
        )
        if result.scalar() is None:
            continue

        result = await conn.execute(text(f"PRAGMA table_info({table})"))
        existing = {row[1] for row in result.fetchall()}

        for column, ddl in columns.items():
            if column not in existing:
                await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _run_column_migrations(conn)
        # WAL lets readers and writers proceed concurrently; busy_timeout makes
        # concurrent writers wait instead of failing with "database is locked".
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.execute(text("PRAGMA busy_timeout=30000"))
        await conn.execute(text("PRAGMA foreign_keys=ON"))
        # Singleton safety: emotional_state is one row for the whole bot.
        # Stray duplicates (bad maintenance script, bug) raise "Multiple rows
        # were found" on every message and brick the pipeline — keep oldest.
        await conn.execute(
            text(
                "DELETE FROM emotional_state WHERE id NOT IN "
                "(SELECT MIN(id) FROM emotional_state)"
            )
        )


async def close_db():
    await engine.dispose()


async def commit_with_retry(session, attempts: int = 3):
    """Commit with retry on transient SQLite contention.

    Under concurrent writers, SQLite in WAL mode can surface
    "database is locked" even with busy_timeout (the read->write
    transaction upgrade path is NOT retried by busy_timeout). A short
    backoff + retry turns those into non-events instead of lost messages.
    """
    for attempt in range(1, attempts + 1):
        try:
            await session.commit()
            return
        except OperationalError as e:
            if "database is locked" in str(e).lower() and attempt < attempts:
                await asyncio.sleep(0.4 * attempt)
                continue
            raise


@asynccontextmanager
async def get_session():
    async with async_session_factory() as session:
        try:
            yield session
            await commit_with_retry(session)
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_session_sync():
    return async_session_factory()