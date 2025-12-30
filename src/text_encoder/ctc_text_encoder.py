import re
import os
from string import ascii_lowercase

import torch
from tokenizers import Tokenizer


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, tokenizer_path=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii. Ignored if tokenizer_path is provided.
            tokenizer_path (str): path to a saved BPE tokenizer.
        """
        self.tokenizer = None
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
        """
        Raw decoding without CTC.
        """
        if self.tokenizer is not None:
            return self.tokenizer.decode(list(inds), skip_special_tokens=False)
        
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        res_inds = []
        last_ind = -1
        for ind in inds:
            if ind == last_ind:
                continue
            if ind != 0:  # 0 is blank/EMPTY_TOK
                res_inds.append(int(ind))
            last_ind = ind
        
        if self.tokenizer is not None:
            return self.tokenizer.decode(res_inds)
        
        return "".join([self.ind2char[ind] for ind in res_inds])

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

        final_beam = {}
        for (ids_tuple, last_id), (p_nb, p_b) in beam.items():
            if self.tokenizer is not None:
                text = self.tokenizer.decode(list(ids_tuple))
            else:
                text = "".join([self.ind2char[idx] for idx in ids_tuple])
            final_beam[text] = final_beam.get(text, 0.0) + p_nb + p_b

        return sorted(final_beam.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
