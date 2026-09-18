import json
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
TRAIN_PATH = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/clause_train.json"
TEST_PATH = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/clause_test.json"
LABEL_MAP_PATH = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/label_mapping.json"
OUTPUT_DIR = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/clause_classifier_model"

MAX_LENGTH = 256
NUM_EPOCHS = 3
BATCH_SIZE = 16

def load_json_dataset(path: str):
    with open (path, 'r', encoding="utf-8") as f:
        rows = json.load(f)
    return Dataset.from_list(rows)

def main():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    num_labels = len(label_mapping)
    id2label = {int(k): v for k, v in label_mapping.items()}
    label2id = {v: int(k) for k, v in label_mapping.items()}
    print(f"Loaded {num_labels} clause-type labels")

    print("Loading train/test data...")
    train_ds = load_json_dataset(TRAIN_PATH)
    test_ds = load_json_dataset(TEST_PATH)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    print(f"Loading tokenizer & model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH
        )

    print("Tokenizing...")
    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    train_ds = train_ds.rename_column("label_id", "labels")
    test_ds = test_ds.rename_column("label_id", "labels")
    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in keep_cols])
    train_ds.set_format("torch")
    test_ds.set_format("torch")

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("\nFinal evaluation on test set:")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"\nSaving best model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done. Model ready for evaluation.")


if __name__ == "__main__":
    main()