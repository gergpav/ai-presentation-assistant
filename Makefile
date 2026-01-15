.PHONY: help install build up down restart logs clean test migrate dev-backend dev-frontend dev-worker dev dev-stop deploy

# Переменные
DOCKER_COMPOSE = docker-compose
PYTHON = python

help: ## Показать справку по командам
	@echo "Доступные команды:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================
# Установка и настройка
# ============================================

install: install-backend install-frontend ## Установить все зависимости

install-backend: ## Установить зависимости backend
	@echo "📦 Установка зависимостей backend..."
	python -m venv venv || true
	. venv/bin/activate || . venv/Scripts/activate || true
	pip install -r requirements.txt

install-frontend: ## Установить зависимости frontend
	@echo "📦 Установка зависимостей frontend..."
	cd frontend && npm install

# ============================================
# Docker команды
# ============================================

build: ## Собрать Docker образ
	$(DOCKER_COMPOSE) build

up: ## Запустить все сервисы
	$(DOCKER_COMPOSE) up -d
	@echo "✅ Сервисы запущены."
	@echo "🌐 Frontend: http://localhost:80"
	@echo "🔧 Backend API: http://localhost:8001"
	@echo "📊 API Docs: http://localhost:8001/docs"

down: ## Остановить все сервисы
	$(DOCKER_COMPOSE) down

restart: down up ## Перезапустить все сервисы

logs: ## Показать логи всех сервисов
	$(DOCKER_COMPOSE) logs -f

logs-app: ## Показать логи приложения
	$(DOCKER_COMPOSE) logs -f app

logs-worker: ## Показать логи worker
	$(DOCKER_COMPOSE) logs -f worker

logs-db: ## Показать логи базы данных
	$(DOCKER_COMPOSE) logs -f db

# ============================================
# Разработка (без Docker)
# ============================================

dev-backend: ## Запустить backend локально (без Docker)
	@echo "🚀 Запуск backend..."
	. venv/bin/activate || . venv/Scripts/activate || true
	python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001

dev-frontend: ## Запустить frontend локально
	@echo "🚀 Запуск frontend..."
	cd frontend && npm run dev

dev-worker: ## Запустить worker локально
	@echo "🚀 Запуск worker..."
	. venv/bin/activate || . venv/Scripts/activate || true
	python -m app.workers.runner

dev: ## Запустить все сервисы локально (backend, worker, frontend)
	@echo "🚀 Запуск всех сервисов..."
	@echo "📝 Backend будет запущен на http://127.0.0.1:8001"
	@echo "👷 Worker будет запущен в отдельном окне"
	@echo "🌐 Frontend будет запущен в отдельном окне"
	@echo ""
	@echo "⚠️  Для остановки закройте все окна или нажмите Ctrl+C"
	@echo ""
ifeq ($(OS),Windows_NT)
	@echo "🔧 Запуск Backend..."
	@start "Backend - AI Presentation Assistant" cmd /k "venv\Scripts\activate && python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001"
	@timeout /t 3 /nobreak >nul
	@echo "👷 Запуск Worker..."
	@start "Worker - AI Presentation Assistant" cmd /k "venv\Scripts\activate && python -m app.workers.runner"
	@timeout /t 2 /nobreak >nul
	@echo "🌐 Запуск Frontend..."
	@start "Frontend - AI Presentation Assistant" cmd /k "cd frontend && npm run dev"
	@echo ""
	@echo "✅ Все сервисы запущены в отдельных окнах!"
	@echo "📊 Backend API: http://127.0.0.1:8001/docs"
	@echo "🌐 Frontend: http://localhost:5173"
else
	@echo "🔧 Запуск Backend в фоне..."
	@. venv/bin/activate && python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001 > /tmp/backend.log 2>&1 &
	@echo $$! > /tmp/backend.pid
	@sleep 2
	@echo "👷 Запуск Worker в фоне..."
	@. venv/bin/activate && python -m app.workers.runner > /tmp/worker.log 2>&1 &
	@echo $$! > /tmp/worker.pid
	@sleep 1
	@echo "🌐 Запуск Frontend..."
	@echo ""
	@echo "✅ Backend и Worker запущены в фоне!"
	@echo "📊 Backend API: http://127.0.0.1:8001/docs"
	@echo "📝 Логи Backend: tail -f /tmp/backend.log"
	@echo "📝 Логи Worker: tail -f /tmp/worker.log"
	@echo ""
	@echo "Для остановки: make dev-stop"
	@cd frontend && npm run dev
endif

dev-stop: ## Остановить все локальные сервисы
ifeq ($(OS),Windows_NT)
	@echo "🛑 Остановка сервисов..."
	@taskkill /FI "WindowTitle eq Backend - AI Presentation Assistant*" /T /F >nul 2>&1 || true
	@taskkill /FI "WindowTitle eq Worker - AI Presentation Assistant*" /T /F >nul 2>&1 || true
	@taskkill /FI "WindowTitle eq Frontend - AI Presentation Assistant*" /T /F >nul 2>&1 || true
	@echo "✅ Сервисы остановлены"
else
	@echo "🛑 Остановка сервисов..."
	@kill `cat /tmp/backend.pid 2>/dev/null` 2>/dev/null || true
	@kill `cat /tmp/worker.pid 2>/dev/null` 2>/dev/null || true
	@rm -f /tmp/backend.pid /tmp/worker.pid /tmp/backend.log /tmp/worker.log
	@echo "✅ Сервисы остановлены"
endif

# ============================================
# База данных
# ============================================

migrate: ## Применить миграции базы данных
	. venv/bin/activate || . venv/Scripts/activate || true
	python -m alembic upgrade head

migrate-create: ## Создать новую миграцию (использовать: make migrate-create NAME=description)
	. venv/bin/activate || . venv/Scripts/activate || true
	python -m alembic revision --autogenerate -m "$(NAME)"

migrate-docker: ## Применить миграции в Docker контейнере
	$(DOCKER_COMPOSE) exec app python -m alembic upgrade head

# ============================================
# Тестирование
# ============================================

test: ## Запустить тесты
	. venv/bin/activate || . venv/Scripts/activate || true
	pytest app/tests/ -v

test-docker: ## Запустить тесты в Docker
	$(DOCKER_COMPOSE) exec app pytest app/tests/ -v

# ============================================
# Развертывание
# ============================================

deploy: build up migrate-docker ## Полное развертывание (сборка + запуск + миграции)
	@echo "✅ Развертывание завершено"

# ============================================
# Очистка
# ============================================

clean: ## Очистить временные файлы и кэш
	@echo "🧹 Очистка..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	rm -rf frontend/node_modules/.cache 2>/dev/null || true
	rm -rf frontend/dist 2>/dev/null || true

clean-docker: ## Очистить Docker образы и volumes
	$(DOCKER_COMPOSE) down -v
	docker system prune -f

clean-all: clean clean-docker ## Полная очистка (включая Docker)

# ============================================
# Утилиты
# ============================================

shell-app: ## Открыть shell в app контейнере
	$(DOCKER_COMPOSE) exec app /bin/bash

shell-db: ## Открыть psql в базе данных
	$(DOCKER_COMPOSE) exec db psql -U postgres -d ai_presentation

health: ## Проверить здоровье сервисов
	@echo "🏥 Проверка здоровья сервисов..."
	@curl -s http://localhost:8001/health | python -m json.tool || echo "❌ Backend недоступен"
	@curl -s http://localhost:80/ > /dev/null && echo "✅ Frontend доступен" || echo "❌ Frontend недоступен"

setup: install migrate ## Полная настройка проекта (установка + миграции)

status: ## Показать статус сервисов
	$(DOCKER_COMPOSE) ps
