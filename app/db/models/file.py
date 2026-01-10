from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import BigInteger, ForeignKey, String, DateTime, func, Integer
from sqlalchemy.dialects.postgresql import ENUM as PGEnum
from app.db.base import Base
from app.db.models.enums import FileKind

class File(Base):
    __tablename__ = "files"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    project_id: Mapped[int | None] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True, nullable=True)

    kind: Mapped[FileKind] = mapped_column(PGEnum(FileKind, name="file_kind"), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user: Mapped["User"] = relationship(back_populates="files")
    project: Mapped["Project | None"] = relationship(back_populates="files")
