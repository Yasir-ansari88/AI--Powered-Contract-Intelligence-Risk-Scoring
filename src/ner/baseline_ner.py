import json
import spacy

nlp = spacy.load("en_core_web_sm")

RELEVANT_LABELS = {"ORG", "DATE", "MONEY", "GPE", "LAW"}

def extract_entities(text: str) -> list:

    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in RELEVANT_LABELS:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_Char,
                }
            )
    return entities

def run_baseline_on_dataset(input_path: str = "cuad_processed.json",
                            output_path: str = "cuad_with_entites.json",
                            limit: int = 20):

    with open (input_path, "r", encoding= "utf-8") as f:
        contracts = json.load(f)

    subset = contracts[:limit]
    print(f"Running baseline NER on {len(subset)} contracts")

    for i, contract in enumerate(subset):
        text_sample = contract["text"][:1000_000]
        contract["baseline_entities"] = extract_entities(text_sample)
        print(f"[{i+1} / {len(subset)}] {contract["title"][:50]}..."
              f"{len(contract["baseline_entities"])} entities found")

    with open (output_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)

    print(f"\n Saved Baseline Results to {output_path}")

if __name__ == "__main__":
    run_baseline_on_dataset()