# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT


import numpy as np
from torch.utils.data import Dataset
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from pathlib import Path
from tape.tokenizers import TAPETokenizer
import torch
from tape.datasets import dataset_factory, pad_sequences, LMDBDataset
from copy import copy
import random 
import lmdb
import pickle as pkl

class LMDBDatasetForAntibody(LMDBDataset):
    """Creates a dataset from an lmdb file."""

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):
        """Inits dataset
        
        Args:
            data_file (Union[str, Path]): Path to lmdb file.
            in_memory (bool, optional): Whether to load the full dataset into memory.
                Default: False.
        """
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False, subdir=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))
 
        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

class MaskedLanguageModelingDataset(Dataset):
    """Creates the Masked Language Modeling Dataset Class"""

    def __init__(self,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 maxlen: int = 512, 
                 **kwargs):
        """Inits dataset
        """

        super().__init__()

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        """Turns list of samples into batch that can be fed to a model

        Args:
            batch (List[Any]): A list of samples to be turned into a batch.

        Returns:
            A batch corresponding to the given list of samples
        """
        input_ids, input_mask, lm_label_ids = tuple(zip(*batch))

        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        # ignore_index is -1
        lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))

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

class PfamLanguageModelingDataset(MaskedLanguageModelingDataset):
    def __init__(self,
                 data_path: Union[str, Path],
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 maxlen: int = 512, 
                 in_memory: bool = False,
                 **kwargs):
        """Inits dataset
            Args:
                data_path (Union[str, Path]): Path to tape data root.
                in_memory (bool, optional): Whether to load the full dataset into memory.
                    Default: False.
        """

        super().__init__(tokenizer, maxlen)
        self.data = LMDBDataset(data_path, in_memory)

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

        return masked_token_ids, input_mask, labels


class AntibodyLanguageModelingDataset(MaskedLanguageModelingDataset):
    """Antibody dataset for Bert. 
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 maxlen: int = 512, 
                 in_memory: bool = False,
                 **kwargs):
        """Inits dataset
            Args:
                data_path (Union[str, Path]): Path to tape data root.
                in_memory (bool, optional): Whether to load the full dataset into memory.
                    Default: False.
        """

        super().__init__(tokenizer, maxlen)
        self.data = LMDBDatasetForAntibody(data_path, in_memory)

    def __len__(self) -> int:
        """Returns dataset length"""
        return len(self.data)

    def __getitem__(self, index):
        """Returns sample using given index

        Args:
            index (int): Dataset index
        """
        item = self.data[index]
        if len(item["seq"]) > self.maxlen:
            item["seq"] = self.slide_long_sequence(item["seq"], self.maxlen)

        tokens = self.tokenizer.tokenize(item['seq'])
        tokens = self.tokenizer.add_special_tokens(tokens)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        input_mask = np.ones_like(masked_token_ids)

        masked_token_ids = np.array(self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        return masked_token_ids, input_mask, labels
