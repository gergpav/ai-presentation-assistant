from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.auth_service import get_current_user
from app.db.models.user import User
from app.db.models.job import Job

router = APIRouter(tags=["jobs"])

class JobOut(BaseModel):
    id: int
    status: str
    progress: int | None = None
    result_file_id: int | None = None
    error_message: str | None = None

@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(
    job_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(select(Job).where(Job.id == job_id, Job.user_id == user.id))
    job = res.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Job not found")

    return JobOut(
        id=job.id,
        status=str(job.status),
        progress=getattr(job, "progress", None),
        result_file_id=getattr(job, "result_file_id", None),
        error_message=getattr(job, "error_message", None),
    )

