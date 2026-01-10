from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import BigInteger, ForeignKey, String, Integer, Text, DateTime, func, Index
from sqlalchemy.dialects.postgresql import ENUM as PGEnum
from app.db.base import Base
from app.db.models.enums import SlideVisualType, SlideStatus

class Slide(Base):
    __tablename__ = "slides"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True, nullable=False)

    position: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="Слайд")
    visual_type: Mapped[SlideVisualType] = mapped_column(PGEnum(SlideVisualType, name="slide_visual_type"), nullable=False)
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[SlideStatus] = mapped_column(PGEnum(SlideStatus, name="slide_status"), nullable=False, default=SlideStatus.draft)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    project: Mapped["Project"] = relationship(back_populates="slides")
    contents: Mapped[list["SlideContent"]] = relationship(back_populates="slide", cascade="all, delete-orphan")
    documents: Mapped[list["SlideDocument"]] = relationship(back_populates="slide", cascade="all, delete-orphan")
    jobs: Mapped[list["Job"]] = relationship(back_populates="slide")

    __table_args__ = (
        Index("ix_slides_project_position", "project_id", "position", unique=True),
    )
