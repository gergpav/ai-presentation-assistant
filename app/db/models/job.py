from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import BigInteger, ForeignKey, DateTime, func, Integer, Text
from sqlalchemy.dialects.postgresql import ENUM as PGEnum
from app.db.base import Base
from app.db.models.enums import JobType, JobStatus

class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True, nullable=True)
    slide_id: Mapped[int | None] = mapped_column(ForeignKey("slides.id", ondelete="CASCADE"), index=True, nullable=True)

    type: Mapped[JobType] = mapped_column(PGEnum(JobType, name="job_type"), nullable=False)
    status: Mapped[JobStatus] = mapped_column(PGEnum(JobStatus, name="job_status"), nullable=False, default=JobStatus.queued)

    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    result_file_id: Mapped[int | None] = mapped_column(ForeignKey("files.id", ondelete="SET NULL"), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    user: Mapped["User"] = relationship(back_populates="jobs")
    project: Mapped["Project | None"] = relationship(back_populates="jobs")
    slide: Mapped["Slide | None"] = relationship(back_populates="jobs")
    result_file: Mapped["File | None"] = relationship()
