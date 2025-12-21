import re
from string import ascii_lowercase

import torch


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

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
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        res = []
        last_ind = -1
        for ind in inds:
            if ind == last_ind:
                continue
            if ind != 0:  # 0 is EMPTY_TOK
                res.append(self.ind2char[int(ind)])
            last_ind = ind
        return "".join(res)

    def ctc_beam_search(self, probs_for_beam, beam_size=10) -> list:
        beam = {("", self.EMPTY_TOK): (0.0, 1.0)}

        for probs in probs_for_beam:
            new_beam = {}
            for i, p in enumerate(probs):
                p = p.item()
                if p == 0:
                    continue

                char = self.ind2char[i]
                for (text, last_char), (p_nb, p_b) in beam.items():
                    if char == self.EMPTY_TOK:
                        n_text, n_last = text, char
                        p_nb_new, p_b_new = new_beam.get((n_text, n_last), (0.0, 0.0))
                        new_beam[(n_text, n_last)] = (p_nb_new, p_b_new + p * (p_nb + p_b))
                    elif char == last_char:
                        n_text, n_last = text, char
                        p_nb_new, p_b_new = new_beam.get((n_text, n_last), (0.0, 0.0))
                        new_beam[(n_text, n_last)] = (p_nb_new + p * p_nb, p_b_new)

                        p_nb_new, p_b_new = new_beam.get((n_text, n_last), (0.0, 0.0))
                        new_beam[(n_text, n_last)] = (p_nb_new + p * p_b, p_b_new)
                    else:
                        n_text, n_last = text + char, char
                        p_nb_new, p_b_new = new_beam.get((n_text, n_last), (0.0, 0.0))
                        new_beam[(n_text, n_last)] = (p_nb_new + p * (p_nb + p_b), p_b_new)

            beam = dict(
                sorted(new_beam.items(), key=lambda x: sum(x[1]), reverse=True)[:beam_size]
            )

        final_beam = {}
        for (text, last_char), (p_nb, p_b) in beam.items():
            final_beam[text] = final_beam.get(text, 0.0) + p_nb + p_b

        return sorted(final_beam.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
