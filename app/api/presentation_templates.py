import shutil
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from app.core.presentation_analyzer import analyze_template

router = APIRouter()
logger = logging.getLogger(__name__)

# Папка для хранения шаблонов
TEMPLATES_DIR = Path("templates")
TEMPLATES_DIR.mkdir(exist_ok=True)


# Хранилище шаблонов в памяти
templates_store = {}


@router.post("/upload")
async def upload_template(file: UploadFile = File(...), template_name: str = "Пользовательский шаблон"):
    """Загрузка PowerPoint шаблона"""
    if not file.filename.lower().endswith('.pptx'):
        raise HTTPException(status_code=400, detail="Только .pptx файлы поддерживаются")

    try:
        # Генерируем ID для шаблона
        template_id = str(uuid.uuid4())

        # Сохраняем файл
        file_path = TEMPLATES_DIR / f"{template_id}.pptx"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Анализируем шаблон
        template_info = analyze_template(file_path)

        # Сохраняем в хранилище
        templates_store[template_id] = {
            "id": template_id,
            "name": template_name,
            "filename": file.filename,
            "file_path": str(file_path),
            "slides_count": template_info.slides_count,
            "layouts": template_info.layouts
        }

        return {
            "template_id": template_id,
            "name": template_name,
            "layouts": template_info.layouts,
            "message": "Шаблон успешно загружен"
        }

    except Exception as e:
        logger.error(f"Ошибка загрузки шаблона: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки шаблона: {str(e)}")


@router.get("/")
async def list_templates():
    """Список загруженных шаблонов"""
    return list(templates_store.values())


@router.delete("/delete/{template_id}")
async def delete_template(template_id: str):
    """Удаление шаблона"""
    if template_id not in templates_store:
        raise HTTPException(status_code=404, detail="Шаблон не найден")

    try:
        # Удаляем файл
        file_path = TEMPLATES_DIR / f"{template_id}.pptx"
        if file_path.exists():
            file_path.unlink()

        # Удаляем из хранилища
        del templates_store[template_id]

        return {"message": "Шаблон удален"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка удаления: {str(e)}")