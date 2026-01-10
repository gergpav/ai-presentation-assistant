from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import BigInteger, ForeignKey, String, DateTime, func, Text
from app.db.base import Base

class SlideDocument(Base):
    __tablename__ = "slide_documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    slide_id: Mapped[int] = mapped_column(ForeignKey("slides.id", ondelete="CASCADE"), index=True, nullable=False)

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)

    parsed_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunks_count: Mapped[int | None] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    slide: Mapped["Slide"] = relationship(back_populates="documents")
