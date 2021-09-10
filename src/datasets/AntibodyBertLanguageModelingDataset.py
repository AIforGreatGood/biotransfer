# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import  List, Any, Dict, Union, Tuple
from copy import copy
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from tape.tokenizers import TAPETokenizer
import torch
from tape.datasets import pad_sequences
from tape.datasets import dataset_factory
import random
import os,lmdb
import pickle as pkl

class LMDBDatasetForAntibody(Dataset):
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

    def __len__(self) -> int:
        """Returns dataset length"""
        return self._num_examples

    def __getitem__(self, index: int):
        """Returns sample using given index

        Args:
            index (int): Dataset index
        """
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item

class AntibodyBertLanguageModelingDataset(Dataset):
    """Antibody dataset for Bert. Old version, features like percent reduction
       are deprecated in favor of pytorch-lightning features.
    """

    def __init__(self, split: str, data_path: Union[str, Path], in_memory: bool=False, maxlen=200, percent_reduction=1, **kwargs):
        super().__init__()
        print('args',locals())
        self.tokenizer = TAPETokenizer(vocab='iupac')
        data_file = f'oas_heavy_{split}.lmdb'
        data_path = os.path.join(data_path, data_file)
        data_path = Path(data_path)
        #data_file = f'oas_light_{split}.lmdb'
        self.data = LMDBDatasetForAntibody(data_path, in_memory)

        self.maxlen = maxlen
        self.percent_reduction = percent_reduction
        assert self.percent_reduction > 0 and self.percent_reduction <= 1
        self.count = 0
        if self.percent_reduction < 1:
            full_indices = np.arange(start=0, stop=len(self.data))
            self.reduced_indices = np.random.choice(full_indices, size=int(self.percent_reduction*len(self.data)), replace=False)

    def __len__(self) -> int:
        """Returns dataset length"""
        return int(self.percent_reduction*len(self.data))

    def __getitem__(self, index):
        """Returns sample using given index

        Args:
            index (int): Dataset index
        """
        real_index = self.index_lookup(index)
        item = self.data[real_index]
        if len(item["seq"]) > self.maxlen:
            item["seq"] = self.slide_long_sequence(item["seq"], self.maxlen)

        tokens = self.tokenizer.tokenize(item['seq'])
        tokens = self.tokenizer.add_special_tokens(tokens)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        input_mask = np.ones_like(masked_token_ids)

        masked_token_ids = np.array(self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        return masked_token_ids, input_mask, labels

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

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        """Turns list of samples into batch that can be fed to a model

        Args:
            batch (List[Any]): A list of samples to be turned into a batch.

        Returns:
            A batch corresponding to the given list of samples
        """
        input_ids, input_mask, lm_label_ids = tuple(zip(*batch))

        torch_inputs = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))

        torch_labels = torch.from_numpy(pad_sequences(lm_label_ids, -1))
        if self.count < 0:
            torch.set_printoptions(profile="full")
            print('input_ids', input_ids)
            self.count += 1
        return {'input_ids': torch_inputs,
            'input_mask': input_mask,
            'targets': torch_labels}

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
                    # 80% chance for random change to mask token
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
