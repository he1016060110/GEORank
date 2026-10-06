# Fork modification (he1016060110, 2026-10-06): isolate explicit test database event loops without changing production pooling.
"""
数据库连接与 Session 管理
"""
import os
import re
from collections.abc import Mapping

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.core.config import settings


_PROTECTED_DATABASE_NAMES = {"georank", "postgres", "template0", "template1"}
_TEST_DATABASE_MARKER = re.compile(r"(?:^|[-_])(?:test|ci)(?:$|[-_0-9])", re.IGNORECASE)


def _engine_options(database_url: str, environment: Mapping[str, str]) -> dict:
    """Keep production pooling; isolated test loops must not share asyncpg connections.

    The test package validates TEST_DATABASE_URL before bootstrapping POSTGRES_*
    settings. Do not reinterpret TEST_DATABASE_URL or change the app's database
    target here: pooling changes only for an explicitly selected, matching test DB.
    A .env default alone is not an explicit integration-test admission.
    """
    options = {
        "pool_size": 20,
        "max_overflow": 10,
        "pool_recycle": 3600,
        "pool_pre_ping": True,
    }
    configured_name = environment.get("POSTGRES_DB", "").strip()
    if not configured_name:
        return options
    parsed_url = make_url(database_url)
    database_name = (parsed_url.database or "").strip()
    if (
        parsed_url.drivername != "postgresql+asyncpg"
        or configured_name != database_name
        or database_name.lower() in _PROTECTED_DATABASE_NAMES
        or not _TEST_DATABASE_MARKER.search(database_name)
    ):
        return options

    explicit_url = environment.get("TEST_DATABASE_URL", "").strip()
    if explicit_url:
        try:
            test_url = make_url(explicit_url)
        except Exception:
            return options
        if test_url.drivername in {"postgres", "postgresql"}:
            test_url = test_url.set(drivername="postgresql+asyncpg")
        if test_url.port is None:
            test_url = test_url.set(port=5432)
        if (
            test_url.drivername != "postgresql+asyncpg"
            or test_url.query
            or not test_url.host
            or not test_url.username
            or test_url.password is None
            or test_url != parsed_url
        ):
            return options

    return {"poolclass": NullPool, "pool_pre_ping": True}


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    **_engine_options(settings.DATABASE_URL, os.environ),
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    """FastAPI 依赖注入 — 获取数据库 Session"""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
