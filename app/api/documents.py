# app/api/documents.py
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File as UploadFileField
from pydantic import BaseModel
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import FileKind
from app.db.session import get_db
from app.services.auth_service import get_current_user
from app.db.models.user import User
from app.db.models.project import Project
from app.db.models.slide import Slide
from app.db.models.file import File  # твоя модель "files"
from app.db.models.slide_document import SlideDocument  # твоя связка "slide_documents"

router = APIRouter(tags=["documents"])

STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)


class SlideDocumentOut(BaseModel):
    id: int
    filename: str | None = None
    size_bytes: int | None = None
    mime_type: str | None = None


async def _ensure_slide_owner(db: AsyncSession, slide_id: int, user_id: int) -> Slide:
    res = await db.execute(
        select(Slide)
        .join(Project, Project.id == Slide.project_id)
        .where(Slide.id == slide_id, Project.user_id == user_id)
    )
    slide = res.scalar_one_or_none()
    if not slide:
        raise HTTPException(status_code=404, detail="Slide not found")
    return slide


@router.post("/slides/{slide_id}/documents", response_model=SlideDocumentOut, status_code=201)
async def upload_slide_document(
    slide_id: int,
    upload: UploadFile = UploadFileField(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    slide = await _ensure_slide_owner(db, slide_id, user.id)

    # 1) сохраняем файл на диск
    ext = Path(upload.filename or "").suffix
    safe_name = f"{uuid.uuid4().hex}{ext}"
    dst_path = STORAGE_DIR / safe_name

    content = await upload.read()
    dst_path.write_bytes(content)

    # 2) создаём запись File
    f = File(
        user_id=user.id,
        project_id=slide.project_id,
        kind=FileKind.document,
        filename=upload.filename,
        storage_path=str(dst_path),
        size_bytes=len(content),
    )
    db.add(f)
    await db.commit()
    await db.refresh(f)

    # 3) привязка к слайду
    sd = SlideDocument(
        slide_id=slide.id,
        filename=upload.filename or "file",
        mime_type=upload.content_type or "application/octet-stream",
        storage_path=str(dst_path),
        parsed_text=None,
        chunks_count=None,
    )
    db.add(sd)
    await db.commit()
    await db.refresh(sd)

    return SlideDocumentOut(
        id=sd.id,
        filename=sd.filename,
        size_bytes=len(content),
    )


@router.get("/slides/{slide_id}/documents", response_model=list[SlideDocumentOut])
async def list_slide_documents(
    slide_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    slide = await _ensure_slide_owner(db, slide_id, user.id)

    res = await db.execute(
        select(SlideDocument)
        .where(SlideDocument.slide_id == slide.id)
        .order_by(SlideDocument.id.asc())
    )

    out = []
    for sd in res.scalars().all():
        # size_bytes у SlideDocument нет — можно посчитать по файлу (опционально)
        out.append(SlideDocumentOut(
            id=sd.id,
            filename=sd.filename,
            mime_type=sd.mime_type,
            size_bytes=None,
        ))
    return out


@router.delete("/slides/{slide_id}/documents/{doc_id}", status_code=204)
async def delete_slide_document(
    slide_id: int,
    doc_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    slide = await _ensure_slide_owner(db, slide_id, user.id)

    res = await db.execute(
        select(SlideDocument)
        .where(SlideDocument.id == doc_id, SlideDocument.slide_id == slide.id)
    )
    sd = res.scalar_one_or_none()
    if not sd:
        raise HTTPException(status_code=404, detail="Document link not found")

    # удаляем только связь; сам файл можно оставить (или удалить отдельно)
    await db.execute(delete(SlideDocument).where(SlideDocument.id == sd.id))
    await db.commit()
    return None
