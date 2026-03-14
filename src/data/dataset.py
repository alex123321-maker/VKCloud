import pandas as pd
import torch
from torch.utils.data import Dataset


TARGET_LABELS = {"negative": 0, "neutral": 1, "positive": 2}


class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row["text"])
        label = int(row["label"])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def normalize_rusentiment_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    if "sentiment" in normalized.columns:
        normalized = normalized.rename(columns={"sentiment": "label"})

    if "text" not in normalized.columns or "label" not in normalized.columns:
        raise ValueError(
            "CSV должен содержать столбцы 'text' и 'label' или 'sentiment'."
        )

    normalized["label"] = normalized["label"].astype(str).str.lower().str.strip()
    normalized = normalized[normalized["label"].isin(TARGET_LABELS.keys())].copy()
    normalized["label"] = normalized["label"].map(TARGET_LABELS)
    normalized["text"] = normalized["text"].astype(str)
    normalized = normalized.dropna(subset=["text", "label"])

    if normalized.empty:
        raise ValueError(
            "После фильтрации не осталось записей с метками positive/neutral/negative."
        )

    return normalized


def load_data(data_path, label_map=None):
    """Загружает CSV и приводит метки к трехклассовой схеме тональности."""
    dataframe = pd.read_csv(data_path)
    dataframe = normalize_rusentiment_labels(dataframe)
    current_label_map = label_map or TARGET_LABELS
    return dataframe, current_label_map
