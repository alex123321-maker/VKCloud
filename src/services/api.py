import json
import logging
import os
from pathlib import Path

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

async function analyze() {
  const text = document.getElementById("inputText").value.trim();
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
      body: JSON.stringify({text: text})
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || "Ошибка сервера");
    }
    const data = await resp.json();
    const cls = data.sentiment || "unknown";
    resultDiv.className = "result " + cls;
    sentimentLabel.textContent = SENTIMENT_RU[cls] || cls;
    confidenceLabel.textContent = "Уверенность модели: " + (data.confidence * 100).toFixed(1) + "%";
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
</script>
</body>
</html>
"""

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/rubert_sentiment_model"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None
inv_label_map = {}


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


def load_model() -> None:
    global model, tokenizer, inv_label_map

    ensure_local_model()

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()

        label_map_path = MODEL_PATH / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, "r", encoding="utf-8") as file:
                label_map = json.load(file)
            inv_label_map = {int(value): key for key, value in label_map.items()}

        logger.info("Модель успешно загружена")
    except Exception as exc:
        logger.error("Не удалось загрузить модель: %s", exc)
        model = None
        tokenizer = None
        inv_label_map = {}


load_model()


@app.get("/", response_class=HTMLResponse)
async def index():
    """Главная страница с веб-интерфейсом для анализа тональности."""
    return _INDEX_HTML


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float


@app.post("/predict", response_model=SentimentResponse)
async def predict(request: TextRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

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
    )


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None, "device": str(device)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
