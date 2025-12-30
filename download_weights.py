import os
import argparse
import gdown

def download_file(file_id, output_path):
    if not file_id:
        print(f"Skipping download for {output_path} (no ID provided)")
        return
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--tokenizer_id", type=str, default=None)
    args = parser.parse_args()

    os.makedirs("saved/ConformerV5_bs_16_bpe_1k_other", exist_ok=True)
    
    download_file(args.model_id, "saved/ConformerV5_bs_16_bpe_1k_other/model_best.pth")
    download_file(args.tokenizer_id, "src/text_encoder/bpe_tokenizer.json")

