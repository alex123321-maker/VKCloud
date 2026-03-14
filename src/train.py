import json
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.data.dataset import SentimentDataset, load_data
from src.storage.s3 import s3_enabled, upload_directory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_demo_dataset(data_path: str) -> None:
    demo_data = pd.DataFrame(
        {
            "text": [
                "Это отличный продукт",
                "Очень плохое качество",
                "В целом нормально",
            ]
            * 100,
            "label": ["positive", "negative", "neutral"] * 100,
        }
    )
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    demo_data.to_csv(data_path, index=False)


def train(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["pretrained_model"],
        num_labels=config["model"]["num_labels"],
    )
    model.to(device)

    data_path = os.path.join(config["paths"]["data_raw"], "train.csv")
    if not os.path.exists(data_path):
        if not config["training"].get("use_dummy_data", False):
            raise FileNotFoundError(
                f"Файл датасета не найден: {data_path}. Поместите туда RuSentiment в формате CSV."
            )

        print("Файл train.csv не найден. Будет создан демонстрационный датасет.")
        prepare_demo_dataset(data_path)

    dataframe, label_map = load_data(data_path)
    print(f"Размер датасета после фильтрации: {len(dataframe)}")
    print(f"Карта меток: {label_map}")

    train_df, val_df = train_test_split(
        dataframe,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["seed"],
        stratify=dataframe["label"],
    )

    train_dataset = SentimentDataset(
        train_df, tokenizer, config["training"]["max_length"]
    )
    val_dataset = SentimentDataset(val_df, tokenizer, config["training"]["max_length"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
    )

    optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
    total_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        print(
            f"Эпоха {epoch + 1}/{config['training']['epochs']} - средний train loss: {avg_train_loss:.4f}"
        )

        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_val_loss += outputs.loss.item()

                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        accuracy = correct / max(total, 1)
        print(f"Валидация - loss: {avg_val_loss:.4f}, accuracy: {accuracy:.4f}")

    model_save_path = os.path.join(config["paths"]["models"], "rubert_sentiment_model")
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    with open(
        os.path.join(model_save_path, "label_map.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(label_map, file, ensure_ascii=False, indent=2)

    print(f"Модель сохранена в {model_save_path}")

    if s3_enabled():
        bucket = os.getenv("S3_BUCKET", config["cloud"]["s3_bucket"])
        prefix = os.getenv("S3_MODEL_PREFIX", config["cloud"]["s3_model_prefix"])
        upload_directory(model_save_path, bucket, prefix)
        print(f"Модель загружена в VK Object Storage: s3://{bucket}/{prefix}")


if __name__ == "__main__":
    train()
