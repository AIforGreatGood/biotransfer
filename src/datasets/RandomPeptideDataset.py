# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from copy import copy
from pathlib import Path
import pickle as pkl
import logging
import random

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform

from tape.tokenizers import TAPETokenizer
from tape.datasets import dataset_factory, pad_sequences

class RandomPeptideDataset(Dataset):
    """Creates the Random Peptide Dataset"""

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac'):
        """Inits dataset"""
        super().__init__()
        if split not in ('train', 'valid'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid', 'holdout']")
        
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        # Load dataset
        self.env = self.load_lmdb(data_path, split)

        # Load number of samples
        with self.env.begin(write=False) as txn:
            self.num_samples = pkl.loads(txn.get(b'num_examples'))

    def load_lmdb(self, directory, split):    
        """ Function to read in LMDB file and print entries for sanity check"""
        directory += "random_peptide_%s.lmdb" % split
        filepath = Path(directory)
        env = lmdb.open(str(filepath), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)
        
        return env

    def __len__(self) -> int:
        """Returns dataset length"""
        return self.num_samples

    def __getitem__(self, index):
        """Returns sample using given index

        Args:
            index (int): Dataset index
        """
        #test read from index: first 10 entries
        with self.env.begin(write=False) as txn:
            item = pkl.loads(txn.get(str(index).encode()))
            item = "".join(item)

        tokens = self.tokenizer.tokenize(item)
        tokens = self.tokenizer.add_special_tokens(tokens)
        token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(tokens), np.int64)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        input_mask = np.ones_like(masked_token_ids)

        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)

        return token_ids, masked_token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        """Turns list of samples into batch that can be fed to a model

        Args:
            batch (List[Any]): A list of samples to be turned into a batch.

        Returns:
            A batch corresponding to the given list of samples
        """
        input_ids, masked_input_ids, input_masks, mask_targets = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        masked_input_ids = torch.from_numpy(pad_sequences(masked_input_ids, 0))
        input_masks = torch.from_numpy(pad_sequences(input_masks, 0))
        mask_targets = torch.from_numpy(pad_sequences(mask_targets, 0))
        return {'input_ids': input_ids,
                'masked_input_ids': masked_input_ids,
                'input_mask': input_masks,
                'targets': mask_targets}

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
                pass

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
