import jsonlines
import difflib
from codebleu import calc_codebleu

def compute_exact_match(jsonl_path):

    total = 0
    correct = 0

    with jsonlines.open(jsonl_path, 'r') as reader:
        for obj in reader:
            actual = obj.get("actual", "").strip()
            label = obj.get("label", "").strip()
            total += 1
            if actual == label:
                correct += 1

    return correct / total if total > 0 else 0.0

def compute_codebleu(jsonl_path, lang="python"):

    references = []
    predictions = []

    with jsonlines.open(jsonl_path, 'r') as reader:
        for obj in reader:
            references.append([obj.get("label", "").strip()])
            predictions.append(obj.get("actual", "").strip())

    return calc_codebleu(references, predictions, lang)

if __name__ == '__main__':
    jsonl_file_path = "data.jsonl"
    exact_match_score = compute_exact_match(jsonl_file_path)
    codebleu_score = compute_codebleu(jsonl_file_path)

    print(f"Exact Match Score: {exact_match_score}")
    print(f"CodeBLEU Score: {codebleu_score}")

