import json
from collections import defaultdict
from datasets import load_dataset

print("Loading CUAD dataset...")
dataset = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)
train_data = dataset["train"]

print(f"Total question-answer rows: {len(train_data)}")
print("Sample row:")
print(train_data[0])

print("\nGrouping rows by contract...")
contracts = defaultdict(lambda: {"contract_id": None, "title": None, "text": None, "clauses": []})

for row in train_data:
    context = row["context"]
    title = row["title"]
    question = row["question"]
    clause_type = question.split("'")[1] if "'" in question else question
    question = row["question"]
    clause_type = question.split("'")[1] if "'" in question else question

    key = title  # each unique title = one contract
    contracts[key]["contract_id"] = key
    contracts[key]["title"] = title
    contracts[key]["text"] = context

    answers = row["answers"]
    for text, start in zip(answers["text"], answers["answer_start"]):
        contracts[key]["clauses"].append({
            "clause_type": clause_type,
            "text": text,
            "start": start,
            "end": start + len(text),
        })

contracts_list = list(contracts.values())
print(f"Total unique contracts: {len(contracts_list)}")
print(f"Example contract has {len(contracts_list[0]['clauses'])} labeled clauses")

output_path = "cuad_processed.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(contracts_list, f, indent=2)

print(f"\nSaved processed dataset to {output_path}")
print("Each record: {contract_id, title, text, clauses: [{clause_type, text, start, end}, ...]}")
