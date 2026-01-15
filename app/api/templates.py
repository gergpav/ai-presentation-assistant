from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models import Template
from app.db.models.user import User
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/templates", tags=["templates"])


@router.post("/upload")
async def upload_template(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # 1) валидация
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is empty")

    if not file.filename.lower().endswith(".pptx"):
        raise HTTPException(status_code=400, detail="Only .pptx files are allowed")

    # 2) папка хранения
    base_dir = Path("storage") / "templates"
    base_dir.mkdir(parents=True, exist_ok=True)

    # 3) формируем безопасное имя на диске
    # чтобы не было коллизий и странных символов — сохраняем как "<ts>_<original_name>"
    safe_name = os.path.basename(file.filename).replace(" ", "_")
    disk_name = f"{int(time.time())}_{safe_name}"
    storage_path = (base_dir / disk_name).resolve()

    # 4) сохраняем файл
    content = await file.read()
    storage_path.write_bytes(content)

    # 5) пишем в БД
    tmpl = Template(
        user_id=user.id,
        filename=file.filename,
        storage_path=str(storage_path),
        meta={
            "content_type": file.content_type,
            "uploaded_at": datetime.now().isoformat(),
        },
    )

    db.add(tmpl)
    await db.commit()
    await db.refresh(tmpl)

    return {
        "id": tmpl.id,
        "user_id": tmpl.user_id,
        "filename": tmpl.filename,
        "storage_path": tmpl.storage_path,
        "created_at": tmpl.created_at.isoformat() if tmpl.created_at else None,
    }
