#!/bin/bash
set -e

echo "🚀 Starting AI Presentation Assistant..."

# Проверяем, запускается ли worker (через аргумент или переменную окружения)
if [ "$1" = "worker" ] || [ "${RUN_WORKER:-false}" = "true" ]; then
    echo "👷 Starting worker..."
    # Ожидание базы данных
    echo "⏳ Waiting for database to be ready..."
    sleep 2
    # Применение миграций для worker
    echo "📦 Running database migrations..."
    python -m alembic upgrade head || echo "⚠️  Migration failed, continuing..."
    # Запуск worker
    exec python -m app.workers.runner
fi

# Обычный запуск приложения (app контейнер)
echo "🌐 Starting application..."

# Ожидание базы данных
echo "⏳ Waiting for database to be ready..."
sleep 2

# Применение миграций
echo "📦 Running database migrations..."
python -m alembic upgrade head || echo "⚠️  Migration failed, continuing..."

# Запуск nginx для фронтенда в фоне
echo "🌐 Starting nginx for frontend..."
nginx

# Запуск FastAPI
echo "🔧 Starting FastAPI backend..."
exec python -m uvicorn app.main:app --reload --host 0.0.0.0 --port ${PORT:-8001}
