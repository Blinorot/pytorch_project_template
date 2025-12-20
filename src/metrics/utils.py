import numpy as np


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 1.0 if len(predicted_text) > 0 else 0.0
    return levenshtein_distance(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    target_words = target_text.split()
    predicted_words = predicted_text.split()
    if len(target_words) == 0:
        return 1.0 if len(predicted_words) > 0 else 0.0
    return levenshtein_distance(target_words, predicted_words) / len(target_words)
