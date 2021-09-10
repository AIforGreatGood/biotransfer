# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT


import numpy as np
#from tape.datasets import MaskedLanguageModelingDataset
from torch.utils.data import Dataset
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from pathlib import Path
from tape.tokenizers import TAPETokenizer
import torch
from tape.datasets import dataset_factory, pad_sequences
from copy import copy
import random 

class MaskedLanguageModelingDataset(Dataset):
    """Creates the Masked Language Modeling Pfam Dataset"""

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False,
                 **kwargs):
        """Inits dataset

        Args:
            data_path (Union[str, Path]): Path to tape data root.
            split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
            in_memory (bool, optional): Whether to load the full dataset into memory.
                Default: False.
        """

        super().__init__()
        if split not in ('train', 'valid', 'holdout'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid', 'holdout']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'pfam_{split}.lmdb' #change back to pfam
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        """Returns dataset length"""
        return len(self.data)

    def __getitem__(self, index):
        """Returns sample using given index

        Args:
            index (int): Dataset index
        """
        item = self.data[index]
        tokens = self.tokenizer.tokenize(item['primary'])
        tokens = self.tokenizer.add_special_tokens(tokens)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        input_mask = np.ones_like(masked_token_ids)

        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)

        return masked_token_ids, input_mask, labels, item['clan'], item['family']

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        """Turns list of samples into batch that can be fed to a model

        Args:
            batch (List[Any]): A list of samples to be turned into a batch.

        Returns:
            A batch corresponding to the given list of samples
        """
        input_ids, input_mask, lm_label_ids, clan, family = tuple(zip(*batch))

        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        # ignore_index is -1
        lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))
        clan = torch.LongTensor(clan)  # type: ignore
        family = torch.LongTensor(family)  # type: ignore

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': lm_label_ids}

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        """Apply a bert mask to the given token sequence

        Args:
            tokens (List[str]): a list of tokens

        Returns:
            A tuple of masked tokens and labels. The masked tokens are 
            the original tokens with random tokens masked or changed. The
            labels should indicate which positions were changed and how.
        """
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1

        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                continue

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)

                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token
                    token = self.tokenizer.convert_id_to_token(
                        random.randint(0, self.tokenizer.vocab_size - 1))
                else:
                    # 10% chance to keep current token
                    pass

                masked_tokens[i] = token

        return masked_tokens, labels

class TruncMaskedLanguageModelingDataset(MaskedLanguageModelingDataset):
    """Creates the Masked Language Modeling Pfam Dataset"""

    def __init__(self, maxlen, percent_reduction=1, **kwargs):
        """Inits dataset

        Args:
            data_path (Union[str, Path]): Path to tape data root.
            split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
            in_memory (bool, optional): Whether to load the full dataset into memory.
                Default: False.
        """
        super().__init__(**kwargs)

        self.maxlen = maxlen
        self.percent_reduction = percent_reduction
        assert self.percent_reduction > 0 and self.percent_reduction <= 1

        if self.percent_reduction < 1:
            full_indices = np.arange(start=0, stop=len(self.data))
            self.reduced_indices = np.random.choice(full_indices, size=int(self.percent_reduction*len(self.data)), replace=False)

    def __getitem__(self, index):
        """Returns sample using given index

        Args:
            index (int): Dataset index
        """
        real_index = self.index_lookup(index)
        item = self.data[real_index]

        if len(item["primary"]) > self.maxlen:
            item["primary"] = self.slide_long_sequence(item["primary"], self.maxlen)

        tokens = self.tokenizer.tokenize(item['primary'])
        tokens = self.tokenizer.add_special_tokens(tokens)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        input_mask = np.ones_like(masked_token_ids)

        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)

        return masked_token_ids, input_mask, labels, item['clan'], item['family']

    def __len__(self):
        """Returns dataset length"""
        return int(self.percent_reduction*len(self.data))

    def index_lookup(self, index):
        """Converts given reduced dataset index to full dataset index

        Args:
            index (int): Reduced dataset index

        Returns:
            The real dataset index corresponding to the reduced index
        """
        if self.percent_reduction == 1:
            return index
        else:
            return self.reduced_indices[index]

    def slide_long_sequence(self, seq, max_seqlen):
        """Clip sequence lengths longer than a max length

        Args:
            seq: The sequence to be clipped.
            max_seqlen: The max sequence length allowed

        Returns:
            The clipped sequence
        """
        start = np.random.randint(len(seq) - max_seqlen + 1)
        end = start + max_seqlen
        seq = seq[start:end]

        return seq
