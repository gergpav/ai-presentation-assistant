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
# Используем production build с правильным API URL (относительный путь для работы через nginx)
ARG VITE_API_BASE_URL=/api
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
    git \
    fonts-dejavu \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копирование requirements и установка зависимостей Python
COPY requirements.txt .
# Обновляем pip перед установкой зависимостей
RUN pip install --upgrade pip setuptools wheel

# Установка PyTorch с CUDA поддержкой (для GPU)
# Используем CUDA 12.1, который совместим с CUDA Toolkit 13.0
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Устанавливаем остальные зависимости (torch уже установлен, поэтому будет пропущен)
# Это решает проблему с несовпадающими хешами пакетов (например, hf-xet)
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Примечание: kernels может показать предупреждение о невозможности установить CPU ядро
# Это не критично - bitsandbytes будет работать без оптимизации, просто медленнее

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
