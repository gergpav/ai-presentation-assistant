#!/bin/bash
set -e

echo "🚀 Starting AI Presentation Assistant..."

# Ожидание базы данных
# Docker Compose управляет зависимостями через depends_on,
# но добавим небольшую задержку для гарантии
echo "⏳ Waiting for database to be ready..."
sleep 2

# Применение миграций
echo "📦 Running database migrations..."
alembic upgrade head || echo "⚠️  Migration failed, continuing..."

# Запуск nginx для фронтенда в фоне
echo "🌐 Starting nginx for frontend..."
nginx

# Запуск FastAPI
echo "🔧 Starting FastAPI backend..."
exec python -m uvicorn app.main:app --reload --host 0.0.0.0 --port ${PORT:-8001}
