# Multi-stage build для объединения фронтенда и бекенда

# ============================================
# Stage 1: Build Frontend
# ============================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Копирование package файлов
COPY frontend/package*.json ./

# Установка зависимостей
RUN npm install

# Копирование исходного кода фронтенда
COPY frontend/ .

# Сборка фронтенда
# Используем production build с правильным API URL
ARG VITE_API_BASE_URL=http://localhost:8001
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

# ============================================
# Stage 2: Backend + Frontend
# ============================================
FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копирование requirements и установка зависимостей Python
COPY requirements.txt .
# Обновляем pip перед установкой зависимостей
RUN pip install --upgrade pip setuptools wheel
# Устанавливаем зависимости с --upgrade для обновления хешей зависимостей
# Это решает проблему с несовпадающими хешами пакетов (например, hf-xet)
# bitsandbytes устанавливаем отдельно, так как может требовать специальной сборки
RUN pip install --no-cache-dir --upgrade -r requirements.txt || \
    (pip install --no-cache-dir --upgrade -r requirements.txt --no-deps && \
     pip install --no-cache-dir bitsandbytes>=0.41.0)

# Копирование кода бекенда
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Копирование собранного фронтенда из Stage 1
COPY --from=frontend-builder /app/frontend/dist /usr/share/nginx/html

# Копирование конфигурации nginx
COPY nginx.conf /etc/nginx/sites-available/default
RUN ln -sf /etc/nginx/sites-available/default /etc/nginx/sites-enabled/default

# Создание директории для storage
RUN mkdir -p /app/storage

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8001

# Порты: 8001 для FastAPI, 80 для nginx (frontend)
EXPOSE 8001 80

# Скрипт запуска всех сервисов
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
