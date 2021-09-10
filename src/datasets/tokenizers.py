# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""A collection of tokenizers modified from Tape's original tokenizer"""

from tape.tokenizers import TAPETokenizer, IUPAC_VOCAB
import numpy as np
from collections import OrderedDict

UNIPROT21_VOCAB = OrderedDict([
                                ("A", 0),
                                ("R", 1),
                                ("N", 2),
                                ("D", 3),
                                ("C", 4),
                                ("Q", 5),
                                ("E", 6),
                                ("G", 7),
                                ("H", 8),
                                ("I", 9),
                                ("L", 10),
                                ("K", 11),
                                ("M", 12),
                                ("F", 13),
                                ("P", 14),
                                ("S", 15),
                                ("T", 16),
                                ("W", 17),
                                ("Y", 18),
                                ("V", 19),
                                ("X", 20),
                                ("O", 11),
                                ("U", 4),
                                ("B", 20),
                                ("Z", 20)])

class AffinityTokenizer(TAPETokenizer):
        def __init__(self, vocab = "iupac"):
                super().__init__(vocab)
        def tokenize(self, text: str):
                tokens = []
                for x in text:
                        if x in IUPAC_VOCAB.keys():
                                tokens.append(x)
                        else:
                             	tokens.append('<unk>')
                return tokens

class AffinityUniprot21Tokenizer(AffinityTokenizer):
	def __init__(self, vocab: str='uniprot21'):
		if vocab == 'uniprot21':
			self.vocab = UNIPROT21_VOCAB
		self.tokens = list(self.vocab.keys())
		self._vocab_type = vocab
	def encode(self, text:str) -> np.ndarray:
		tokens = self.tokenize(text)
		token_ids = self.convert_tokens_to_ids(tokens)
		return np.array(token_ids, np.int64)
