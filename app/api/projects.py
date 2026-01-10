# app/api/projects.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models.user import User
from app.db.models.project import Project
from app.db.models.enums import AudienceType
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    audience_type: AudienceType
    template_id: int | None = None

    @field_validator("template_id", mode="before")
    @classmethod
    def normalize_template_id(cls, v):
        # фронт часто шлёт 0 как "не выбрано"
        if v in (0, "0", "", "null", None):
            return None
        return v


class ProjectOut(BaseModel):
    id: int
    title: str
    audience_type: AudienceType
    template_id: int | None = None


@router.get("", response_model=list[ProjectOut])
async def list_projects(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(Project).where(Project.user_id == user.id).order_by(Project.updated_at.desc())
    )
    projects = res.scalars().all()
    return [
        ProjectOut(
            id=p.id,
            title=p.title,
            audience_type=p.audience_type,
            template_id=p.template_id,
        )
        for p in projects
    ]


@router.post("", response_model=ProjectOut, status_code=201)
async def create_project(
    payload: ProjectCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = Project(
        user_id=user.id,
        title=payload.title,
        audience_type=payload.audience_type,
        template_id=payload.template_id,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)

    return ProjectOut(
        id=project.id,
        title=project.title,
        audience_type=project.audience_type,
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
