import json
from scipy.stats import ttest_ind

target_label = 'A'

path_pairs = [
    ("generated_predictions_clean.jsonl", "generated_predictions_watermark.jsonl")
]
for clean_path, poison_path in path_pairs:
    suspected_sequence = []
    with open(clean_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            suspected_sequence.append(1 if item['predict'] == target_label else 0)

    original_sequence = []
    with open(poison_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            original_sequence.append(1 if item['predict'] == target_label else 0)

    t_stat, p_value = ttest_ind(suspected_sequence, original_sequence)


    print(f"T statistic: {t_stat:.3f}")
    print(f"P value:     {p_value:.3f}")
    print(suspected_sequence)
    print(original_sequence)
