from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, DateTime, func, BigInteger
from app.db.base import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    login: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    projects: Mapped[list["Project"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    templates: Mapped[list["Template"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    jobs: Mapped[list["Job"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    files: Mapped[list["File"]] = relationship(back_populates="user", cascade="all, delete-orphan")
