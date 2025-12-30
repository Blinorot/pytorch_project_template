import re
import os
from string import ascii_lowercase

import torch
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, tokenizer_path=None, lm_hf_model=None, alpha=0.5, beta=0.1, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii. Ignored if tokenizer_path is provided.
            tokenizer_path (str): path to a saved BPE tokenizer.
            lm_hf_model (str): HuggingFace model name for rescoring (e.g., 'gpt2' or 'distilgpt2').
            alpha (float): LM weight for rescoring.
            beta (float): length penalty.
        """
        self.tokenizer = None
        self.hf_model = None
        self.hf_tokenizer = None
        self.alpha = alpha
        self.beta = beta

        if tokenizer_path is not None:
            if os.path.exists(tokenizer_path):
                self.tokenizer = Tokenizer.from_file(tokenizer_path)
                vocab = self.tokenizer.get_vocab()
                self.vocab = [None] * len(vocab)
                for tok, ind in vocab.items():
                    self.vocab[ind] = tok
                
                self.ind2char = {i: tok for i, tok in enumerate(self.vocab)}
                self.char2ind = {tok: i for i, tok in enumerate(self.vocab)}
            else:
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        else:
            if alphabet is None:
                alphabet = list(ascii_lowercase + " ")
            self.alphabet = alphabet
            self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
            self.ind2char = dict(enumerate(self.vocab))
            self.char2ind = {v: k for k, v in self.ind2char.items()}

        if lm_hf_model is not None:
            print(f"Loading HF LM: {lm_hf_model} for rescoring...")
            self.hf_tokenizer = GPT2Tokenizer.from_pretrained(lm_hf_model)
            self.hf_model = GPT2LMHeadModel.from_pretrained(lm_hf_model)
            self.hf_model.eval()
            if torch.cuda.is_available():
                self.hf_model.to("cuda")
            print(f"Loaded HF LM from {lm_hf_model}")

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        if self.tokenizer is not None:
            return torch.Tensor(self.tokenizer.encode(text).ids).unsqueeze(0)
        
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(list(inds), skip_special_tokens=False)
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        res_inds = []
        last_ind = -1
        for ind in inds:
            if ind == last_ind:
                continue
            if ind != 0:  # 0 is blank
                res_inds.append(int(ind))
            last_ind = ind
        
        if self.tokenizer is not None:
            return self.tokenizer.decode(res_inds)
        return "".join([self.ind2char[ind] for ind in res_inds])

    def get_hf_lm_score(self, texts):
        if self.hf_model is None:
            return [0.0] * len(texts)
        
        scores = []
        device = next(self.hf_model.parameters()).device
        
        with torch.no_grad():
            for text in texts:
                if not text.strip():
                    scores.append(-100.0)
                    continue
                
                inputs = self.hf_tokenizer(text, return_tensors="pt").to(device)
                labels = inputs["input_ids"]
                outputs = self.hf_model(**inputs, labels=labels)
                log_likelihood = -outputs.loss.item() * labels.size(1)
                scores.append(log_likelihood)
        return scores

    def ctc_beam_search(self, probs_for_beam, beam_size=10) -> list:
        beam = {((), -1): (0.0, 1.0)} 

        for probs in probs_for_beam:
            new_beam = {}
            active_indices = torch.where(probs > 1e-5)[0]
            for i in active_indices:
                i = i.item()
                p = probs[i].item()

                for (ids_tuple, last_id), (p_nb, p_b) in beam.items():
                    if i == 0:  # blank
                        n_ids, n_last = ids_tuple, i
                        p_nb_new, p_b_new = new_beam.get((n_ids, n_last), (0.0, 0.0))
                        new_beam[(n_ids, n_last)] = (p_nb_new, p_b_new + p * (p_nb + p_b))
                    elif i == last_id:
                        n_ids, n_last = ids_tuple, i
                        p_nb_new, p_b_new = new_beam.get((n_ids, n_last), (0.0, 0.0))
                        new_beam[(n_ids, n_last)] = (p_nb_new + p * p_nb, p_b_new)
                        p_nb_new, p_b_new = new_beam.get((n_ids, n_last), (0.0, 0.0))
                        new_beam[(n_ids, n_last)] = (p_nb_new + p * p_b, p_b_new)
                    else:
                        n_ids, n_last = ids_tuple + (i,), i
                        p_nb_new, p_b_new = new_beam.get((n_ids, n_last), (0.0, 0.0))
                        new_beam[(n_ids, n_last)] = (p_nb_new + p * (p_nb + p_b), p_b_new)

            beam = dict(
                sorted(new_beam.items(), key=lambda x: sum(x[1]), reverse=True)[:beam_size]
            )

        # 2. Final scores from CTC
        hypotheses = []
        for (ids_tuple, last_id), (p_nb, p_b) in beam.items():
            if self.tokenizer is not None:
                text = self.tokenizer.decode(list(ids_tuple))
            else:
                text = "".join([self.ind2char[idx] for idx in ids_tuple])
            
            p_ctc = p_nb + p_b
            ctc_score = torch.log(torch.tensor(p_ctc) + 1e-10).item()
            hypotheses.append({"text": text, "ctc_score": ctc_score})

        if self.hf_model is not None:
            texts = [h["text"] for h in hypotheses]
            lm_scores = self.get_hf_lm_score(texts)
            
            for i in range(len(hypotheses)):
                final_score = hypotheses[i]["ctc_score"] + self.alpha * lm_scores[i] + self.beta * len(hypotheses[i]["text"].split())
                hypotheses[i]["final_score"] = final_score
            
            hypotheses = sorted(hypotheses, key=lambda x: x["final_score"], reverse=True)
        else:
            hypotheses = sorted(hypotheses, key=lambda x: x["ctc_score"], reverse=True)

        return [(h["text"], h.get("final_score", h["ctc_score"])) for h in hypotheses]

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
