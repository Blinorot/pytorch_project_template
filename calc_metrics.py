import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer


def main(target_dir, predicted_dir):
    target_path = Path(target_dir)
    predicted_path = Path(predicted_dir)

    cers = []
    wers = []

    for pred_file in predicted_path.glob("*.txt"):
        target_file = target_path / pred_file.name
        if not target_file.exists():
            print(f"Warning: Target file {target_file} not found for {pred_file}")
            continue

        with pred_file.open() as f:
            pred_text = f.read().strip().lower()
        with target_file.open() as f:
            target_text = f.read().strip().lower()

        cers.append(calc_cer(target_text, pred_text))
        wers.append(calc_wer(target_text, pred_text))

    if not cers:
        print("No matching transcription files found.")
        return

    print(f"Total files: {len(cers)}")
    print(f"Average CER: {sum(cers) / len(cers):.4f}")
    print(f"Average WER: {sum(wers) / len(wers):.4f}")


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

