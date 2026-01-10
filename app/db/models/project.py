from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import BigInteger, ForeignKey, String, DateTime, func
from sqlalchemy.dialects.postgresql import ENUM as PGEnum
from app.db.base import Base
from app.db.models.enums import AudienceType

class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    audience_type: Mapped[AudienceType] = mapped_column(
        PGEnum(AudienceType, name="audience_type"),
        nullable=False,
    )

    template_id: Mapped[int | None] = mapped_column(ForeignKey("templates.id", ondelete="SET NULL"), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    user: Mapped["User"] = relationship(back_populates="projects")
    template: Mapped["Template | None"] = relationship(back_populates="projects")

    slides: Mapped[list["Slide"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    jobs: Mapped[list["Job"]] = relationship(back_populates="project")
    files: Mapped[list["File"]] = relationship(back_populates="project")
