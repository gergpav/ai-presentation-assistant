from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.parser import extract_text_from_file
from app.core.embeddings import document_index
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


IMAGES_DIR = Path("storage") / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(('.pptx', '.docx', '.pdf', '.xlsx')):
            raise HTTPException(status_code=400, detail="Неподдерживаемый формат")

        document_data = await extract_text_from_file(file)
        document_index.add_documents([document_data])
        document_index.build_index()

        return {
            "filename": file.filename,
            "characters": len(document_data["text"]),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Загрузка изображения для слайда.
    Картинка сохраняется на диск, возвращаем её id и путь.
    """
    filename = file.filename or "image"
    ext = filename.lower().rsplit(".", 1)[-1]

    if ext not in ("png", "jpg", "jpeg", "webp"):
        raise HTTPException(status_code=400, detail="Поддерживаются только PNG/JPG/JPEG/WEBP")

    image_id = str(uuid4())
    stored_name = f"{image_id}.{ext}"
    path = IMAGES_DIR / stored_name

    content = await file.read()
    try:
        with open(path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Ошибка сохранения изображения: {e}")
        raise HTTPException(status_code=500, detail="Не удалось сохранить изображение")

    logger.info(f"Загружено изображение {filename} -> {path}")

    return {
        "image_id": image_id,
        "filename": filename,
        "path": str(path),
    }