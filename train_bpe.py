import os
import argparse
from glob import glob
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDecoder

def train_bpe(data_path, vocab_size, save_path):
    files = glob(os.path.join(data_path, "**/*.txt"), recursive=True)
    if not files:
        print("No transcript files found!")
        return

    all_texts = []
    for f in files:
        with open(f, 'r') as reader:
            for line in reader:
                parts = line.strip().split(' ', 1)
                if len(parts) > 1:
                    all_texts.append(parts[1].lower())
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Metaspace(replacement=" ")
    tokenizer.decoder = MetaspaceDecoder(replacement=" ")

    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["[PAD]", "[UNK]"]
    )

    tokenizer.train_from_iterator(all_texts, trainer)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path} with Metaspace support")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default="src/text_encoder/bpe_tokenizer.json")
    args = parser.parse_args()
    train_bpe(args.data_path, args.vocab_size, args.save_path)
