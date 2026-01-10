# app/api/generate.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models.user import User
from app.db.models.slide import Slide
from app.db.models.project import Project
from app.db.models.job import Job
from app.db.models.enums import JobType, JobStatus, SlideStatus
from app.services.auth_service import get_current_user

router = APIRouter(tags=["generate"])


class CreateJobResponse(BaseModel):
    job_id: int


@router.post("/slides/{slide_id}/generate", response_model=CreateJobResponse, status_code=202)
async def generate_slide(
    slide_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # слайд должен принадлежать пользователю
    res = await db.execute(
        select(Slide)
        .join(Project, Project.id == Slide.project_id)
        .where(Slide.id == slide_id, Project.user_id == user.id)
    )
    slide = res.scalar_one_or_none()
    if not slide:
        raise HTTPException(status_code=404, detail="Slide not found")

    if not slide.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    slide.status = SlideStatus.generating

    job = Job(
        user_id=user.id,
        project_id=slide.project_id,
        slide_id=slide.id,
        type=JobType.generate_slide,
        status=JobStatus.queued,
        progress=0,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    return CreateJobResponse(job_id=job.id)

