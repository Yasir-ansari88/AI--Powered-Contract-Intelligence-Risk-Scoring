import json
from sklearn.model_selection  import train_test_Split
from sklearn.preprocessing import LabelEncoder

INPUT_PATH = 'E:\AI- Powered Contract Intelligence & Risk Scoring\cuad_processed.json'
OUTPUT_TRAIN = "/content/clause_train.json"
OUTPUT_TEST = "/content/clause_test.json"
LABEL_MAP_PATH = "/content/label_mapping.json"

MIN_CHARS = 15

def faltten_contracts(contracts: list):

    rows =[]
    for contract in contracts:
        for clause in contract['clause']:
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