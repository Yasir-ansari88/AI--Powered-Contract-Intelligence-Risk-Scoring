import json
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import LabelEncoder

INPUT_PATH = '/content/AI--Powered-Contract-Intelligence-Risk-Scoring/cuad_processed.json'
OUTPUT_TRAIN = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/clause_train.json"
OUTPUT_TEST = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/clause_test.json"
LABEL_MAP_PATH = "/content/AI--Powered-Contract-Intelligence-Risk-Scoring/label_mapping.json"

MIN_CHARS = 15

def faltten_contracts(contracts: list):

    rows =[]
    for contract in contracts:
        for clause in contract['clauses']:
            text = clause["text"].strip()
            if len(text) >= MIN_CHARS:
                rows.append({
                    "text": text,
                    "label": clause["clause_type"]
                })
    return rows

def main ():
    print(f"Loading {INPUT_PATH}...")
    with open (INPUT_PATH, "r", encoding="utf-8") as f:
        contracts = json.load(f)

    rows = faltten_contracts(contracts)
    print(f"Flattened into {len(rows)} labeled clause example")
    print(f"Unique clause type : {len(set(r['label'] for r in rows))}")

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform([r["label"] for r in rows])
    for row, label_id in zip(rows, labels_encoded):
        row["label_id"] = int(label_id)

    label_mapping = {int(i): cls for i, cls in enumerate(label_encoder.classes_)}
    with open (LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)
    print(f"Saved {len(label_mapping)} label classes to {LABEL_MAP_PATH}")

    train_rows, test_rows = train_test_split(
        rows, test_size = 0.15, random_state = 42, stratify = [r["label_id"] for r in rows]
    )

    print(f"Train: {len(train_rows)} | Test : {len(test_rows)}")

    with open (OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        json.dump(train_rows, f, indent=2)

    with open (OUTPUT_TEST, "w", encoding="utf-8") as f:
        json.dump(test_rows, f, indent=2)

    print(f"\n Saved {OUTPUT_TRAIN} and {OUTPUT_TEST}")
    print("Ready for the fine-tunning")

if __name__ == "__main__":
    main()