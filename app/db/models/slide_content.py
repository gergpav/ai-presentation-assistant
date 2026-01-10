from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import BigInteger, ForeignKey, Integer, DateTime, func, JSON
from app.db.base import Base

class SlideContent(Base):
    __tablename__ = "slide_contents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    slide_id: Mapped[int] = mapped_column(ForeignKey("slides.id", ondelete="CASCADE"), index=True, nullable=False)

    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[dict] = mapped_column(JSON, nullable=False)
    llm_meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    slide: Mapped["Slide"] = relationship(back_populates="contents")
