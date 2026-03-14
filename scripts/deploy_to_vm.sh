#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Использование: ./scripts/deploy_to_vm.sh <user@host> <path_to_ssh_key>"
  exit 1
fi

REMOTE_HOST="$1"
SSH_KEY="$2"
REMOTE_DIR="~/rubert-sentiment"

echo "Создание директории на сервере"
ssh -i "$SSH_KEY" "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

echo "Копирование проекта на сервер"
scp -i "$SSH_KEY" -r \
  Dockerfile docker-compose.yml requirements.txt .env.example README.md configs scripts src \
  "$REMOTE_HOST:$REMOTE_DIR"

echo "Подготовка .env на сервере"
ssh -i "$SSH_KEY" "$REMOTE_HOST" "cd $REMOTE_DIR && [ -f .env ] || cp .env.example .env"

echo "Сборка и запуск контейнера"
ssh -i "$SSH_KEY" "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose up -d --build"

echo "Деплой завершен"
