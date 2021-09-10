# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
import time

from .tokenizers import AffinityTokenizer
from tape.registry import registry
from tape import ProteinBertForValuePrediction
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from pathlib import Path
from tape.datasets import dataset_factory
import numpy as np
from tape.datasets import pad_sequences

##Create a DataLoader for the affinity task and register it

#@registry.register_task('affinity_new')
class GiffordDataset(Dataset):
    """Dataset for Gifford paper data"""

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, AffinityTokenizer] = 'iupac',
                 in_memory: bool = False,
                 collate_type: str = "batch_padded",
                 token_encode_type: str = "embed"):
        """Inits dataset"""

        if split not in ('train', 'val', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'val', 'test']")
        if isinstance(tokenizer, str):
            tokenizer = AffinityTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)

        data_file = f'gifford_{split}.json'

        self.data = dataset_factory(data_path / data_file, in_memory)

        self._max_length = None

        # If batch_padded, pads each sample to max length in current batch
        # If full_padded, pads each sample to the max length in dataset
        assert collate_type in ["batch_padded", "full_padded"]
        self.collate_type = collate_type

        assert token_encode_type in ["embed", "one_hot"]
        self.token_encode_type = token_encode_type

        # This is the length of the longest sequence in the whole dataset,
        # including val and test.
        self.max_length = 22

    def __len__(self) -> int:
        """Returns dataset length"""
        return len(self.data)

    def __getitem__(self, index: int):
        """Returns sample from dataset at specific index

        Args:
            index (int): Index of dataset
        """
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['sequences'])
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, float(item['target'])

    def calculate_max_length(self):
            max_length = 0
            for i in range(len(self)):
                current_len = len(self[i][0])
                if current_len > max_length:
                    max_length = current_len

            return max_length

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        """Turns list of samples into batch that can be fed to a model

        Args:
            batch (List[Any]): A list of samples to be turned into a batch.

        Returns:
            A batch corresponding to the given list of samples
        """
        input_ids, input_masks, affinity_true_value = tuple(zip(*batch))

        if self.collate_type == "batch_padded":
            input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
            input_masks = torch.from_numpy(pad_sequences(input_masks, 0))
        else:
            input_ids = [np.pad(input_id, (0, self.max_length - len(input_id)), "constant", constant_values=(0,0)) for input_id in input_ids]
            input_ids = np.stack(input_ids)
            input_ids = torch.from_numpy(input_ids)


            input_masks = [np.pad(input_mask, (0, self.max_length - len(input_mask)), "constant", constant_values=(0,0)) for input_mask in input_masks]
            input_masks = np.stack(input_masks)
            input_masks = torch.from_numpy(input_masks)

        print(self.collate_type)
        print('lynnnnnnn input_ids',input_ids.shape)
        affinity_true_value = torch.FloatTensor(affinity_true_value)  # type: ignore
        affinity_true_value = affinity_true_value.unsqueeze(1)

        if self.token_encode_type == "one_hot":
            input_ids = one_hot(input_ids, len(set(self.tokenizer.vocab.values())))
            input_ids = input_ids.permute(0, 2, 1)
            input_ids = input_ids.type(torch.float32)

        print('lynnnnnnn input_ids permute',input_ids.shape)

        return {'input_ids': input_ids,
                'input_masks': input_masks,
                'targets': affinity_true_value}

#registry.register_task_model('affinity', 'transformer_w_params', ProteinBertForValuePrediction)
