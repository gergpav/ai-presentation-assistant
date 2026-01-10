from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import Template

router = APIRouter(prefix="/templates", tags=["templates"])


@router.post("/upload")
async def upload_template(
    file: UploadFile = File(...),
    user_id: int | None = None,
    db: Session = Depends(get_db),
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
    try:
        with open(storage_path, "wb") as f:
            f.write(file.file.read())
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    # 5) пишем в БД
    tmpl = Template(
        user_id=user_id,
        filename=file.filename,
        storage_path=str(storage_path),
        meta={
            "content_type": file.content_type,
            "uploaded_at": datetime.now().isoformat(),
        },
    )

    db.add(tmpl)
    db.commit()
    db.refresh(tmpl)

    return {
        "id": tmpl.id,
        "user_id": tmpl.user_id,
        "filename": tmpl.filename,
        "storage_path": tmpl.storage_path,
        "created_at": tmpl.created_at.isoformat() if tmpl.created_at else None,
    }
