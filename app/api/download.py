from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.auth_service import get_current_user
from app.db.models.user import User
from app.db.models.file import File

router = APIRouter(tags=["files"])

@router.get("/files/{file_id}/download")
async def download_file(
    file_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(select(File).where(File.id == file_id, File.user_id == user.id))
    f = res.scalar_one_or_none()
    if not f:
        raise HTTPException(404, "File not found")

    path = Path(f.storage_path)
    if not path.exists():
        raise HTTPException(404, "File missing on disk")

    # Content-Disposition будет attachment с filename => браузер скачает
    return FileResponse(
        path=str(path),
        filename=f.filename,
        media_type="application/octet-stream",
    )
