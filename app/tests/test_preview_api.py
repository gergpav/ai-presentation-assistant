import pytest
from httpx import AsyncClient, ASGITransport

from app.api import generate
from app.main import app
from app.core import llm_generator


@pytest.mark.asyncio
async def test_preview_returns_generated_slides(monkeypatch):
    """
    Базовый тест: /presentation/preview должен вернуть SlideExport[].
    LLM мокается и возвращает фиксированный текст.
    """

    def fake_generate_from_prompt(prompt: str, context: str, audience: str):
        return {"content": "• пункт 1\n• пункт 2\n• пункт 3"}

    # Важно: мокаем именно глобальный content_generator в модуле llm_generator
    llm_generator.content_generator.is_loaded = True
    monkeypatch.setattr(llm_generator.content_generator, "generate_from_prompt", fake_generate_from_prompt)

    payload = {
        "audience": "Инвесторы",
        "format": "pptx",
        "template_id": None,
        "slides": [
            {"title": "Тест", "prompt": "Сделай 3 буллета", "images": [], "docs": []}
        ],
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/presentation/preview", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert data[0]["title"] == "Тест"
    assert "пункт" in data[0]["content"]
    assert data[0]["images"] == []


@pytest.mark.asyncio
async def test_preview_timeout_returns_fallback(monkeypatch):
    def slow_generate_from_prompt(prompt: str, context: str, audience: str):
        import time
        while True:
            time.sleep(1)

    llm_generator.content_generator.is_loaded = True
    monkeypatch.setattr(llm_generator.content_generator, "generate_from_prompt", slow_generate_from_prompt)

    monkeypatch.setattr(generate, "LLM_TIMEOUT_SEC", 0.1)

    payload = {
        "audience": "Инвесторы",
        "format": "pptx",
        "template_id": None,
        "slides": [
            {"title": "Тест", "prompt": "Сделай 3 буллета", "images": [], "docs": []}
        ],
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/presentation/preview", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert "Таймаут" in data[0]["content"]


@pytest.mark.asyncio
async def test_preview_requires_title_and_prompt(monkeypatch):
    """
    Валидация: если title или prompt пустые — 400.
    """

    llm_generator.content_generator.is_loaded = True

    payload = {
        "audience": "Инвесторы",
        "format": "pptx",
        "template_id": None,
        "slides": [
            {"title": "", "prompt": "ok", "images": [], "docs": []}
        ],
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/presentation/preview", json=payload)

    assert resp.status_code == 400