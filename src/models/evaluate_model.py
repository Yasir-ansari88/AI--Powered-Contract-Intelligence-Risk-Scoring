import json
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report

MODEL_DIR = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/clause_classifier_model"
TEST_PATH = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/clause_test.json"
LABEL_MAP_PATH = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/label_mapping.json"
OUTPUT_PATH = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/test_predictions_with_confidence.json"

MAX_LENGTH = 256
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.60

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    id2label = {int(k): v for k, v in label_mapping.items()}

    print(f"Loading model from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_rows = json.load(f)
    print(f"Evaluating on {len(test_rows)} test examples")

    all_preds, all_labels, all_confidences = [], [], []
    results = []

    with torch.no_grad():
        for i in range(0, len(test_rows), BATCH_SIZE):
            batch = test_rows[i:i + BATCH_SIZE]
            texts = [r["text"] for r in batch]
            true_ids = [r["label_id"] for r in batch]

            enc = tokenizer(texts, truncation=True, padding=True,
                             max_length=MAX_LENGTH, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            confidences, pred_ids = torch.max(probs, dim=-1)

            for text, true_id, pred_id, conf in zip(texts, true_ids, pred_ids.tolist(), confidences.tolist()):
                flag = "needs_review" if conf < CONFIDENCE_THRESHOLD else "ok"
                results.append({
                    "text": text[:200],
                    "true_label": id2label[true_id],
                    "predicted_label": id2label[pred_id],
                    "confidence": round(conf, 4),
                    "status": flag,
                })
                all_preds.append(pred_id)
                all_labels.append(true_id)
                all_confidences.append(conf)

            print(f"  Evaluated {min(i + BATCH_SIZE, len(test_rows))}/{len(test_rows)}")

    
    label_names = [id2label[i] for i in sorted(id2label.keys())]
    print("\n=== Per-clause-type classification report ===")
    report = classification_report(
        all_labels, all_preds,
        labels=sorted(id2label.keys()),
        target_names=label_names,
        zero_division=0,
    )
    print(report)

    
    needs_review_count = sum(1 for r in results if r["status"] == "needs_review")
    print(f"\n Confidence threshold = {CONFIDENCE_THRESHOLD} l")
    print(f"{needs_review_count}/{len(results)} predictions flagged 'needs_review' "
          f"({100 * needs_review_count / len(results):.1f}%)")
    print(f"Average confidence: {np.mean(all_confidences):.4f}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed predictions (with confidence + review flags) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
