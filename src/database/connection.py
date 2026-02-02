"""Database connection management."""

import os
from typing import Optional
import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from src.config.settings import settings


Base = declarative_base()


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self):
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None

    def init_db(self):
        """Initialize database engine and session factory."""
        # Sync engine
        self.engine = create_engine(
            settings.database_url.replace("postgresql://", "postgresql+psycopg2://"),
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        # Async engine
        async_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
        self.async_engine = create_async_engine(
            async_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        self.AsyncSessionLocal = sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_pgvector(self):
        """Initialize pgvector extension."""
        async with self.async_engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.commit()

    def create_tables(self):
        """Create all database tables."""
        from src.database.models import DocumentModel, ChunkModel
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self):
        """Get a synchronous database session."""
        return self.SessionLocal()

    async def get_async_session(self):
        """Get an asynchronous database session."""
        async with self.AsyncSessionLocal() as session:
            yield session


# Global database manager instance
db_manager = DatabaseManager()
