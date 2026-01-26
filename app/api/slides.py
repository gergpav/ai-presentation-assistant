# app/api/slides.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, func, delete, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SlideContent
from app.db.session import get_db
from app.db.models.user import User
from app.db.models.project import Project
from app.db.models.slide import Slide
from app.db.models.enums import SlideVisualType, SlideStatus
from app.services.auth_service import get_current_user

router = APIRouter(tags=["slides"])


class SlideCreate(BaseModel):
    title: str = Field(default="Слайд", min_length=1, max_length=255)
    visual_type: SlideVisualType
    prompt: str | None = None


class SlideUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=255)  # Убрано min_length=1, чтобы разрешить пустую строку
    visual_type: SlideVisualType | None = None
    prompt: str | None = None
    position: int | None = None

    @field_validator("position")
    @classmethod
    def validate_position(cls, v):
        if v is not None and v < 1:
            raise ValueError("position must be >= 1")
        return v


class SlideOut(BaseModel):
    id: int
    project_id: int
    position: int
    title: str
    visual_type: SlideVisualType
    prompt: str | None
    status: SlideStatus


class SlideContentUpdate(BaseModel):
    content: str = Field(min_length=1)


async def _ensure_project_owner(db: AsyncSession, project_id: int, user_id: int) -> Project:
    res = await db.execute(select(Project).where(Project.id == project_id, Project.user_id == user_id))
    project = res.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


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


@router.get("/projects/{project_id}/slides", response_model=list[SlideOut])
async def list_slides(
    project_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _ensure_project_owner(db, project_id, user.id)
    res = await db.execute(select(Slide).where(Slide.project_id == project_id).order_by(Slide.position.asc()))
    slides = res.scalars().all()
    return [
        SlideOut(
            id=s.id,
            project_id=s.project_id,
            position=s.position,
            title=s.title,
            visual_type=s.visual_type,
            prompt=s.prompt,
            status=s.status,
        )
        for s in slides
    ]


@router.post("/projects/{project_id}/slides", response_model=SlideOut, status_code=201)
async def create_slide(
    project_id: int,
    payload: SlideCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _ensure_project_owner(db, project_id, user.id)

    # position = max(position)+1
    res = await db.execute(select(func.coalesce(func.max(Slide.position), 0)).where(Slide.project_id == project_id))
    next_pos = int(res.scalar_one()) + 1

    slide = Slide(
        project_id=project_id,
        position=next_pos,
        title=payload.title,
        visual_type=payload.visual_type,
        prompt=payload.prompt,
        status=SlideStatus.draft,
    )
    db.add(slide)
    await db.commit()
    await db.refresh(slide)

    return SlideOut(
        id=slide.id,
        project_id=slide.project_id,
        position=slide.position,
        title=slide.title,
        visual_type=slide.visual_type,
        prompt=slide.prompt,
        status=slide.status,
    )


@router.get("/slides/{slide_id}/content/latest")
async def get_latest_slide_content(
    slide_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # проверяем, что слайд принадлежит пользователю
    slide = await _ensure_slide_owner(db, slide_id, user.id)

    q = (
        select(SlideContent)
        .where(SlideContent.slide_id == slide.id)
        .order_by(desc(SlideContent.version))
        .limit(1)
    )
    res = await db.execute(q)
    sc = res.scalar_one_or_none()
    if not sc:
        return {"slide_id": slide.id, "content": None}

    # content это JSON (dict), извлекаем текст если есть поле text
    content_text = None
    if sc.content:
        if isinstance(sc.content, dict):
            content_text = sc.content.get("text", str(sc.content))
        else:
            content_text = str(sc.content)

    return {
        "slide_id": slide.id,
        "version": sc.version,
        "content": content_text,
        "created_at": sc.created_at.isoformat() if sc.created_at else None,
    }


@router.get("/slides/{slide_id}/image/latest")
async def get_latest_slide_image(
    slide_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Возвращает последнюю сгенерированную картинку для слайда.
    """
    slide = await _ensure_slide_owner(db, slide_id, user.id)

    res = await db.execute(
        select(SlideContent)
        .where(SlideContent.slide_id == slide.id)
        .order_by(desc(SlideContent.version))
        .limit(1)
    )
    sc = res.scalar_one_or_none()
    if not sc or not sc.llm_meta:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = sc.llm_meta.get("generated_image_path")
    if not image_path:
        raise HTTPException(status_code=404, detail="Image not found")

    path = Path(image_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image missing on disk")

    return FileResponse(
        path=str(path),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@router.put("/slides/{slide_id}/content/latest")
async def save_edited_slide_content(
    slide_id: int,
    payload: SlideContentUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Сохраняет вручную отредактированный контент как НОВУЮ запись SlideContent.
    Это нужно для UI: пользователь редактирует текст во вкладке "Контент" и изменения должны сохраняться.
    """
    slide = await _ensure_slide_owner(db, slide_id, user.id)

    # Получаем последнюю версию для инкремента
    res = await db.execute(
        select(SlideContent)
        .where(SlideContent.slide_id == slide.id)
        .order_by(desc(SlideContent.version))
        .limit(1)
    )
    last = res.scalar_one_or_none()
    next_version = (last.version + 1) if last else 1

    # content это JSON (dict), сохраняем как {"text": content}
    sc = SlideContent(
        slide_id=slide.id,
        version=next_version,
        content={"text": payload.content},
    )

    db.add(sc)
    await db.commit()
    await db.refresh(sc)

    # Возвращаем контент в том же формате
    content_text = None
    if sc.content:
        if isinstance(sc.content, dict):
            content_text = sc.content.get("text", str(sc.content))
        else:
            content_text = str(sc.content)

    return {
        "slide_id": slide.id,
        "version": sc.version,
        "content": content_text,
        "created_at": sc.created_at.isoformat() if sc.created_at else None,
    }


@router.get("/slides/{slide_id}/content")
async def list_slide_content_versions(
    slide_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    slide = await _ensure_slide_owner(db, slide_id, user.id)

    res = await db.execute(
        select(SlideContent)
        .where(SlideContent.slide_id == slide.id)
        .order_by(SlideContent.id.desc())
        .limit(20)
    )
    items = res.scalars().all()

    out = []
    for sc in items:
        if hasattr(sc, "content_text"):
            text = sc.content_text
        elif hasattr(sc, "content"):
            text = sc.content
        elif hasattr(sc, "text"):
            text = sc.text
        else:
            text = None

        out.append({
            "id": sc.id,
            "version": getattr(sc, "version", None),
            "content": text,
            "created_at": getattr(sc, "created_at", None),
        })

    return {"slide_id": slide.id, "items": out}


@router.patch("/slides/{slide_id}", response_model=SlideOut)
async def update_slide(
    slide_id: int,
    payload: SlideUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    slide = await _ensure_slide_owner(db, slide_id, user.id)

    # простое обновление полей (перестановку позиций сделаем отдельным endpoint'ом позже)
    if payload.title is not None:
        # Разрешаем пустую строку - устанавливаем её как есть
        slide.title = payload.title if payload.title else ""  # Пустая строка разрешена
    if payload.visual_type is not None:
        slide.visual_type = payload.visual_type
    if payload.prompt is not None:
        slide.prompt = payload.prompt
    if payload.position is not None:
        slide.position = payload.position  # ⚠️ временно, ниже сделаем правильный reorder

    await db.commit()
    await db.refresh(slide)

    return SlideOut(
        id=slide.id,
        project_id=slide.project_id,
        position=slide.position,
        title=slide.title,
        visual_type=slide.visual_type,
        prompt=slide.prompt,
        status=slide.status,
    )


class SlideReorderRequest(BaseModel):
    slide_ids: list[int]  # новый порядок ID слайдов


@router.post("/projects/{project_id}/slides/reorder")
async def reorder_slides(
    project_id: int,
    payload: SlideReorderRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Изменение порядка слайдов"""
    project = await _ensure_project_owner(db, project_id, user.id)

    # Проверяем, что все слайды принадлежат проекту
    res = await db.execute(
        select(Slide).where(Slide.project_id == project_id, Slide.id.in_(payload.slide_ids))
    )
    slides = {s.id: s for s in res.scalars().all()}
    if len(slides) != len(payload.slide_ids):
        raise HTTPException(status_code=400, detail="Some slides not found or don't belong to project")

    # Обновляем позиции
    for position, slide_id in enumerate(payload.slide_ids, start=1):
        slides[slide_id].position = position

    await db.commit()
    return {"message": "Slides reordered successfully"}


@router.delete("/slides/{slide_id}", status_code=204)
async def delete_slide(
    slide_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    slide = await _ensure_slide_owner(db, slide_id, user.id)
    await db.execute(delete(Slide).where(Slide.id == slide.id))
    await db.commit()
    return None

