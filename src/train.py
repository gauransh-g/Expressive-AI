"""
Transformer Summarization Model Training & Fine-Tuning Module.
Fine-tunes BART / T5 Seq2Seq architectures on the multi-domain conversation dataset with PyTorch.
"""

import argparse
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import load_config
from src.dataset_generator import generate_sample_conversations
from src.preprocessing import preprocess_text
from src.utils import get_logger, load_json, resolve_device

logger = get_logger("TrainSummarizer")


class SummarizationDataset(Dataset):
    """PyTorch Dataset for Seq2Seq summarization fine-tuning."""

    def __init__(self, data: List[Dict], tokenizer, max_input_len: int = 512, max_target_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = preprocess_text(item.get("text", ""))
        target_text = item.get("summary") or input_text[: min(len(input_text), 200)]

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = targets["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }


def train_model(
    data_file: str = "data/raw/sample_conversations.json",
    model_name: str = "facebook/bart-large-cnn",
    output_dir: str = "models/fine_tuned_bart",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 3e-5,
    max_samples: int = 50,
    val_split: float = 0.15,
):
    """
    Fine-tunes the Seq2Seq Transformer model on the provided dataset with validation tracking.
    """
    device = resolve_device("auto")
    logger.info(f"Starting training on device: {device}")

    if not os.path.exists(data_file):
        logger.info("Dataset file not found. Generating dataset first...")
        generate_sample_conversations(num_samples=100, output_path=data_file)

    raw_data = load_json(data_file)
    if max_samples and max_samples < len(raw_data):
        raw_data = raw_data[:max_samples]

    val_count = max(1, int(len(raw_data) * val_split))
    train_data = raw_data[:-val_count]
    val_data = raw_data[-val_count:]

    logger.info(f"Loaded {len(train_data)} train samples and {len(val_data)} validation samples.")
    logger.info(f"Loading base model and tokenizer: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    train_dataset = SummarizationDataset(train_data, tokenizer)
    val_dataset = SummarizationDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    print("\n" + "=" * 65)
    print(">> STARTING TRANSFORMER SUMMARIZATION FINE-TUNING")
    print(f">> Model: {model_name} | Epochs: {num_epochs} | Train: {len(train_data)} | Val: {len(val_data)}")
    print("=" * 65)

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")

        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        logger.info(f"Epoch {epoch} Completed - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Checkpoint if best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out_path))
            tokenizer.save_pretrained(str(out_path))
            logger.info(f">> Saved best checkpoint with Val Loss: {avg_val_loss:.4f}")

    print("\n" + "=" * 65)
    print(f">> FINE-TUNING COMPLETED! Best Val Loss: {best_val_loss:.4f} | Weights: {output_dir}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Transformer Summarization Model")
    parser.add_argument("--data", type=str, default="data/raw/sample_conversations.json")
    parser.add_argument("--model", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--output", type=str, default="models/fine_tuned_bart")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()

    train_model(
        data_file=args.data,
        model_name=args.model,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
