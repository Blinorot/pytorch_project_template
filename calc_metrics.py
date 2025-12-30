import argparse
from pathlib import Path

from src.metrics.utils import levenshtein_distance


def main(target_dir, predicted_dir):
    target_path = Path(target_dir)
    predicted_path = Path(predicted_dir)

    total_cer_dist = 0
    total_cer_len = 0
    total_wer_dist = 0
    total_wer_len = 0
    file_count = 0

    for pred_file in predicted_path.glob("*.txt"):
        target_file = target_path / pred_file.name
        if not target_file.exists():
            print(f"Warning: Target file {target_file} not found for {pred_file}")
            continue

        with pred_file.open() as f:
            pred_text = f.read().strip().lower()
        with target_file.open() as f:
            target_text = f.read().strip().lower()

        total_cer_dist += levenshtein_distance(target_text, pred_text)
        total_cer_len += len(target_text)

        target_words = target_text.split()
        pred_words = pred_text.split()
        total_wer_dist += levenshtein_distance(target_words, pred_words)
        total_wer_len += len(target_words)
        
        file_count += 1

    if file_count == 0:
        print("No matching transcription files found.")
        return

    print(f"Total files: {file_count}")

    final_cer = total_cer_dist / total_cer_len if total_cer_len > 0 else 0
    final_wer = total_wer_dist / total_wer_len if total_wer_len > 0 else 0
    
    print(f"Standard CER: {final_cer:.4f}")
    print(f"Standard WER: {final_wer:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--target_dir", required=True, help="Path to ground truth transcriptions"
    )
    parser.add_argument(
        "-p", "--predicted_dir", required=True, help="Path to model predictions"
    )
    args = parser.parse_args()
    main(args.target_dir, args.predicted_dir)
