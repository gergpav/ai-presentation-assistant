from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Literal

from app.db.session import get_db
from app.db.models.user import User
from app.db.models.project import Project
from app.db.models.job import Job
from app.db.models.enums import JobType, JobStatus
from app.services.auth_service import get_current_user

router = APIRouter(tags=["export"])

class ExportRequest(BaseModel):
    format: Literal["pptx", "pdf"]

class ExportResponse(BaseModel):
    job_id: int

@router.post("/projects/{project_id}/export", response_model=ExportResponse)
async def export_project(
    project_id: int,
    payload: ExportRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Проверяем, что проект принадлежит пользователю
    res = await db.execute(
        select(Project).where(Project.id == project_id, Project.user_id == user.id)
    )
    project = res.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    job = Job(
        user_id=user.id,
        project_id=project_id,
        type=JobType.export_pptx if payload.format == "pptx" else JobType.export_pdf,
        status=JobStatus.queued,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    return ExportResponse(job_id=job.id)


