FROM python:3.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Установка Python-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода проекта
COPY . .

# Создание директории для моделей
RUN mkdir -p models

# Открытие порта API
EXPOSE 8000

# Запуск HTTP API
CMD ["uvicorn", "src.services.api:app", "--host", "0.0.0.0", "--port", "8000"]
