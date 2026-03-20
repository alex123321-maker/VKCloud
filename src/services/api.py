import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.storage.s3 import download_prefix, s3_enabled


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API анализа тональности")


# ---------- HTML-страница ----------

_INDEX_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Анализ тональности текста</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f7fa; color: #333; display: flex;
    justify-content: center; padding: 40px 16px; min-height: 100vh;
  }
  .card {
    background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,.08);
    padding: 32px; max-width: 600px; width: 100%;
  }
  h1 { font-size: 1.5rem; margin-bottom: 8px; }
  .subtitle { color: #666; font-size: .9rem; margin-bottom: 24px; }
  .field-label { display: block; margin-bottom: 8px; color: #444; font-size: .9rem; }
  select {
    width: 100%; margin-bottom: 14px; padding: 10px 12px; font-size: .95rem;
    border: 1px solid #ddd; border-radius: 8px; background: #fff;
  }
  select:focus { outline: none; border-color: #4a90d9; }
  textarea {
    width: 100%; min-height: 120px; padding: 12px; font-size: 1rem;
    border: 1px solid #ddd; border-radius: 8px; resize: vertical;
    font-family: inherit; transition: border-color .2s;
  }
  textarea:focus { outline: none; border-color: #4a90d9; }
  .btn {
    margin-top: 16px; padding: 12px 28px; font-size: 1rem;
    background: #4a90d9; color: #fff; border: none; border-radius: 8px;
    cursor: pointer; transition: background .2s;
  }
  .btn:hover { background: #3a7bc8; }
  .btn:disabled { background: #a0c4e8; cursor: not-allowed; }
  .result {
    margin-top: 24px; padding: 20px; border-radius: 8px;
    display: none; animation: fadeIn .3s ease;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
  .result.positive { background: #e8f5e9; border-left: 4px solid #4caf50; }
  .result.negative { background: #fce4ec; border-left: 4px solid #e53935; }
  .result.neutral  { background: #fff8e1; border-left: 4px solid #fbc02d; }
  .result.error    { background: #fce4ec; border-left: 4px solid #e53935; }
  .sentiment-label { font-size: 1.25rem; font-weight: 600; }
  .confidence { color: #555; margin-top: 6px; font-size: .9rem; }
  .error-msg { color: #c62828; }
</style>
</head>
<body>
<div class="card">
  <h1>Анализ тональности текста</h1>
  <p class="subtitle">Введите отзыв на русском языке, и модель определит его тональность (позитивный / нейтральный / негативный).</p>

  <label class="field-label" for="modelSelect">Модель</label>
  <select id="modelSelect"></select>
  <textarea id="inputText" placeholder="Напишите отзыв здесь..."></textarea>
  <button class="btn" id="analyzeBtn" onclick="analyze()">Анализировать</button>

  <div class="result" id="result">
    <div class="sentiment-label" id="sentimentLabel"></div>
    <div class="confidence" id="confidenceLabel"></div>
  </div>
</div>

<script>
const SENTIMENT_RU = {
  positive: "Позитивный",
  negative: "Негативный",
  neutral:  "Нейтральный",
  unknown:  "Неизвестно"
};

async function loadModels() {
  const modelSelect = document.getElementById("modelSelect");
  try {
    const resp = await fetch("/models");
    if (!resp.ok) throw new Error("Не удалось получить список моделей");
    const data = await resp.json();

    modelSelect.innerHTML = "";
    for (const modelName of data.models || []) {
      const option = document.createElement("option");
      option.value = modelName;
      option.textContent = modelName;
      modelSelect.appendChild(option);
    }

    if (data.default_model) {
      modelSelect.value = data.default_model;
    }
  } catch (e) {
    modelSelect.innerHTML = "";
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Нет доступных моделей";
    modelSelect.appendChild(option);
    modelSelect.disabled = true;
  }
}

async function analyze() {
  const text = document.getElementById("inputText").value.trim();
  const selectedModel = document.getElementById("modelSelect").value;
  if (!text) return;

  const btn = document.getElementById("analyzeBtn");
  const resultDiv = document.getElementById("result");
  const sentimentLabel = document.getElementById("sentimentLabel");
  const confidenceLabel = document.getElementById("confidenceLabel");

  btn.disabled = true;
  btn.textContent = "Обработка...";
  resultDiv.style.display = "none";

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text: text, model: selectedModel})
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || "Ошибка сервера");
    }
    const data = await resp.json();
    const cls = data.sentiment || "unknown";
    resultDiv.className = "result " + cls;
    sentimentLabel.textContent = SENTIMENT_RU[cls] || cls;
    confidenceLabel.textContent = "Модель: " + data.model + " • Уверенность: " + (data.confidence * 100).toFixed(1) + "%";
    resultDiv.style.display = "block";
  } catch (e) {
    resultDiv.className = "result error";
    sentimentLabel.innerHTML = '<span class="error-msg">Ошибка: ' + e.message + '</span>';
    confidenceLabel.textContent = "";
    resultDiv.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.textContent = "Анализировать";
  }
}

// Отправка по Ctrl+Enter
document.getElementById("inputText").addEventListener("keydown", function(e) {
  if (e.ctrlKey && e.key === "Enter") analyze();
});

loadModels();
</script>
</body>
</html>
"""

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/rubert_sentiment_model"))
MODELS_ROOT = Path(os.getenv("MODELS_ROOT", str(MODEL_PATH.parent)))
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", MODEL_PATH.name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
available_models: Dict[str, Path] = {}
loaded_models: Dict[
    str, Tuple[AutoModelForSequenceClassification, AutoTokenizer, Dict[int, str]]
] = {}


def ensure_local_model() -> None:
    if MODEL_PATH.exists() and (MODEL_PATH / "config.json").exists():
        return

    if not s3_enabled():
        return

    bucket = os.getenv("S3_BUCKET")
    prefix = os.getenv("S3_MODEL_PREFIX", "models/rubert_sentiment_model")
    if not bucket:
        return

    logger.info("Локальная модель не найдена, начинается загрузка из VK Object Storage")
    download_prefix(bucket, prefix, str(MODEL_PATH))


def discover_models() -> None:
    global available_models

    models: Dict[str, Path] = {}

    if MODELS_ROOT.exists():
        for path in MODELS_ROOT.iterdir():
            if path.is_dir() and (path / "config.json").exists():
                models[path.name] = path

    if MODEL_PATH.exists() and (MODEL_PATH / "config.json").exists():
        models.setdefault(DEFAULT_MODEL_NAME, MODEL_PATH)

    if not models:
        ensure_local_model()
        if MODEL_PATH.exists() and (MODEL_PATH / "config.json").exists():
            models.setdefault(DEFAULT_MODEL_NAME, MODEL_PATH)

    available_models = dict(sorted(models.items()))


def get_default_model_name() -> Optional[str]:
    if DEFAULT_MODEL_NAME in available_models:
        return DEFAULT_MODEL_NAME
    return next(iter(available_models), None)


def load_model(
    model_name: str,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, Dict[int, str]]:
    if model_name in loaded_models:
        return loaded_models[model_name]

    model_path = available_models.get(model_name)
    if model_path is None:
        raise ValueError(f"Модель '{model_name}' не найдена")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    inv_label_map: Dict[int, str] = {}
    label_map_path = model_path / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path, "r", encoding="utf-8") as file:
            label_map = json.load(file)
        inv_label_map = {int(value): key for key, value in label_map.items()}

    loaded_models[model_name] = (model, tokenizer, inv_label_map)
    logger.info("Модель '%s' успешно загружена", model_name)
    return loaded_models[model_name]


discover_models()


@app.get("/", response_class=HTMLResponse)
async def index():
    """Главная страница с веб-интерфейсом для анализа тональности."""
    return _INDEX_HTML


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    model: Optional[str] = None


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    model: str


class ModelsResponse(BaseModel):
    models: list[str]
    default_model: Optional[str]


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    discover_models()
    return ModelsResponse(
        models=list(available_models.keys()),
        default_model=get_default_model_name(),
    )


@app.post("/predict", response_model=SentimentResponse)
async def predict(request: TextRequest):
    discover_models()

    model_name = request.model or get_default_model_name()
    if model_name is None:
        raise HTTPException(status_code=500, detail="Нет доступных моделей")
    if model_name not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Модель '{model_name}' не найдена. Доступные модели: {', '.join(available_models.keys())}",
        )

    try:
        model, tokenizer, inv_label_map = load_model(model_name)
    except Exception as exc:
        logger.error("Не удалось загрузить модель '%s': %s", model_name, exc)
        raise HTTPException(
            status_code=500, detail=f"Не удалось загрузить модель '{model_name}'"
        ) from exc

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_index].item()

    sentiment = inv_label_map.get(predicted_index, "unknown")
    return SentimentResponse(
        text=request.text,
        sentiment=sentiment,
        confidence=confidence,
        model=model_name,
    )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "available_models": list(available_models.keys()),
        "loaded_models": list(loaded_models.keys()),
        "device": str(device),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
