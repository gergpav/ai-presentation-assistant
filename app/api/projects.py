# app/api/projects.py
from fastapi import APIRouter, Depends, HTTPException
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Literal

from app.db.session import get_db
from app.db.models.user import User
from app.db.models.project import Project
from app.db.models.slide import Slide
from app.db.models.slide_content import SlideContent
from app.db.models.slide_document import SlideDocument
from app.db.models.enums import AudienceType, SlideStatus, SlideVisualType
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/projects", tags=["projects"])

# Маппинг типов аудитории: фронтенд использует другие названия
AUDIENCE_MAPPING_FROM_FRONTEND = {
    "executive": AudienceType.management,
    "expert": AudienceType.experts,
    "investor": AudienceType.investors,
}

AUDIENCE_MAPPING_TO_FRONTEND = {
    AudienceType.management: "executive",
    AudienceType.experts: "expert",
    AudienceType.investors: "investor",
}


class ProjectCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    audience: Literal["executive", "expert", "investor"]  # фронтенд использует такие названия
    template_id: int | None = None

    @field_validator("template_id", mode="before")
    @classmethod
    def normalize_template_id(cls, v):
        # фронт часто шлёт 0 как "не выбрано"
        if v in (0, "0", "", "null", None):
            return None
        return v


class ProjectUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    audience: Literal["executive", "expert", "investor"] | None = None


class SlideDocumentOut(BaseModel):
    id: int
    name: str
    type: str
    size: int


class SlideOut(BaseModel):
    id: int
    title: str
    prompt: str | None
    documents: list[SlideDocumentOut]
    generatedContent: str | None
    generatedImageUrl: str | None = None
    isGenerating: bool
    visualType: Literal["text", "chart", "table", "image"]
    status: Literal["pending", "completed", "failed"]


class ProjectOut(BaseModel):
    id: int
    title: str
    audience: Literal["executive", "expert", "investor"]
    slides: list[SlideOut]
    createdAt: str
    updatedAt: str
    template_id: int | None = None


class ProjectListItem(BaseModel):
    id: int
    title: str
    audience: Literal["executive", "expert", "investor"]
    createdAt: str
    updatedAt: str


async def _ensure_project_owner(db: AsyncSession, project_id: int, user_id: int) -> Project:
    """Проверяет, что проект принадлежит пользователю"""
    res = await db.execute(
        select(Project).where(Project.id == project_id, Project.user_id == user_id)
    )
    project = res.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.get("", response_model=list[ProjectListItem])
async def list_projects(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(Project).where(Project.user_id == user.id).order_by(Project.updated_at.desc())
    )
    projects = res.scalars().all()
    return [
        ProjectListItem(
            id=p.id,
            title=p.title,
            audience=AUDIENCE_MAPPING_TO_FRONTEND[p.audience_type],
            createdAt=p.created_at.isoformat(),
            updatedAt=p.updated_at.isoformat(),
        )
        for p in projects
    ]


@router.post("", response_model=ProjectOut, status_code=201)
async def create_project(
    payload: ProjectCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    audience_type = AUDIENCE_MAPPING_FROM_FRONTEND[payload.audience]
    project = Project(
        user_id=user.id,
        title=payload.title,
        audience_type=audience_type,
        template_id=payload.template_id,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)

    return ProjectOut(
        id=project.id,
        title=project.title,
        audience=AUDIENCE_MAPPING_TO_FRONTEND[project.audience_type],
        slides=[],
        createdAt=project.created_at.isoformat(),
        updatedAt=project.updated_at.isoformat(),
        template_id=project.template_id,
    )


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(
    project_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Получить полный проект со всеми слайдами и их содержимым"""
    project = await _ensure_project_owner(db, project_id, user.id)

    # Получаем все слайды проекта
    slides_res = await db.execute(
        select(Slide).where(Slide.project_id == project_id).order_by(Slide.position.asc())
    )
    slides = slides_res.scalars().all()

    slides_out = []
    for slide in slides:
        # Получаем последнее содержимое слайда
        content_res = await db.execute(
            select(SlideContent)
            .where(SlideContent.slide_id == slide.id)
            .order_by(SlideContent.version.desc())
            .limit(1)
        )
        latest_content = content_res.scalar_one_or_none()
        generated_content = None
        generated_image_url = None
        if latest_content and latest_content.content:
            # content это JSON, извлекаем текст если есть поле text
            if isinstance(latest_content.content, dict):
                generated_content = latest_content.content.get("text", str(latest_content.content))
            else:
                generated_content = str(latest_content.content)
        if latest_content and latest_content.llm_meta:
            image_path = latest_content.llm_meta.get("generated_image_path")
            if image_path and Path(image_path).exists():
                # Добавляем версию, чтобы фронтенд обновлял изображение при регенерации
                generated_image_url = f"/api/slides/{slide.id}/image/latest?v={latest_content.version}"

        # Получаем документы слайда
        docs_res = await db.execute(
            select(SlideDocument).where(SlideDocument.slide_id == slide.id).order_by(SlideDocument.id.asc())
        )
        documents = docs_res.scalars().all()

        # Маппинг статусов
        status_mapping = {
            SlideStatus.draft: "pending",
            SlideStatus.generating: "pending",
            SlideStatus.ready: "completed",
            SlideStatus.error: "failed",
        }

        slides_out.append(
            SlideOut(
                id=slide.id,
                title=slide.title,
                prompt=slide.prompt,
                documents=[
                    SlideDocumentOut(
                        id=doc.id,
                        name=doc.filename,
                        type=doc.mime_type or "application/octet-stream",
                        size=0,  # Размер можно добавить позже, если нужно
                    )
                    for doc in documents
                ],
                generatedContent=generated_content,
                generatedImageUrl=generated_image_url,
                isGenerating=slide.status == SlideStatus.generating,
                visualType=slide.visual_type.value,
                status=status_mapping.get(slide.status, "pending"),
            )
        )

    return ProjectOut(
        id=project.id,
        title=project.title,
        audience=AUDIENCE_MAPPING_TO_FRONTEND[project.audience_type],
        slides=slides_out,
        createdAt=project.created_at.isoformat(),
        updatedAt=project.updated_at.isoformat(),
        template_id=project.template_id,
    )


@router.patch("/{project_id}", response_model=ProjectOut)
async def update_project(
    project_id: int,
    payload: ProjectUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Обновить проект (название и/или аудиторию)"""
    project = await _ensure_project_owner(db, project_id, user.id)

    if payload.title is not None:
        project.title = payload.title
    if payload.audience is not None:
        project.audience_type = AUDIENCE_MAPPING_FROM_FRONTEND[payload.audience]

    await db.commit()
    await db.refresh(project)

    # Возвращаем полный проект (нужно переиспользовать логику из get_project)
    # Упрощенно для начала:
    slides_res = await db.execute(
        select(Slide).where(Slide.project_id == project_id).order_by(Slide.position.asc())
    )
    slides = slides_res.scalars().all()
    slides_out = []
    for slide in slides:
        content_res = await db.execute(
            select(SlideContent)
            .where(SlideContent.slide_id == slide.id)
            .order_by(SlideContent.version.desc())
            .limit(1)
        )
        latest_content = content_res.scalar_one_or_none()
        generated_content = None
        generated_image_url = None
        if latest_content and latest_content.content:
            if isinstance(latest_content.content, dict):
                generated_content = latest_content.content.get("text", str(latest_content.content))
            else:
                generated_content = str(latest_content.content)
        if latest_content and latest_content.llm_meta:
            image_path = latest_content.llm_meta.get("generated_image_path")
            if image_path and Path(image_path).exists():
                # Добавляем версию, чтобы фронтенд обновлял изображение при регенерации
                generated_image_url = f"/api/slides/{slide.id}/image/latest?v={latest_content.version}"

        docs_res = await db.execute(
            select(SlideDocument).where(SlideDocument.slide_id == slide.id).order_by(SlideDocument.id.asc())
        )
        documents = docs_res.scalars().all()

        status_mapping = {
            SlideStatus.draft: "pending",
            SlideStatus.generating: "pending",
            SlideStatus.ready: "completed",
            SlideStatus.error: "failed",
        }

        slides_out.append(
            SlideOut(
                id=slide.id,
                title=slide.title,
                prompt=slide.prompt,
                documents=[
                    SlideDocumentOut(
                        id=doc.id,
                        name=doc.filename,
                        type=doc.mime_type or "application/octet-stream",
                        size=0,
                    )
                    for doc in documents
                ],
                generatedContent=generated_content,
                generatedImageUrl=generated_image_url,
                isGenerating=slide.status == SlideStatus.generating,
                visualType=slide.visual_type.value,
                status=status_mapping.get(slide.status, "pending"),
            )
        )

    return ProjectOut(
        id=project.id,
        title=project.title,
        audience=AUDIENCE_MAPPING_TO_FRONTEND[project.audience_type],
        slides=slides_out,
        createdAt=project.created_at.isoformat(),
        updatedAt=project.updated_at.isoformat(),
        template_id=project.template_id,
    )


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # проверяем, что проект принадлежит юзеру
    res = await db.execute(
        select(Project.id).where(Project.id == project_id, Project.user_id == user.id)
    )
    if res.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Project not found")

    await db.execute(delete(Project).where(Project.id == project_id))
    await db.commit()
    return None
