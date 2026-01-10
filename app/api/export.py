from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.db.session import get_db
from app.services.auth_service import get_current_user
from app.db.models.job import Job
from app.db.models.enums import JobType, JobStatus

router = APIRouter(tags=["export"])

class ExportRequest(BaseModel):
    format: str  # "pptx" | "pdf"

class ExportResponse(BaseModel):
    job_id: int

@router.post("/projects/{project_id}/export", response_model=ExportResponse)
async def export_project(
    project_id: int,
    payload: ExportRequest,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if payload.format not in ("pptx", "pdf"):
        raise HTTPException(400, "Invalid format")

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


