# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from copy import copy
from pathlib import Path
import pickle as pkl
import random
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection

import lmdb
from tape.datasets import pad_sequences
from tape.tokenizers import TAPETokenizer
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class RandomPeptideGenerator(IterableDataset):
    """Creates collection of random peptides"""

    def __init__(self,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 shortest_length=5,
                 longest_length=10,
                 seed=0,
                 **kwargs):
        """Inits dataset"""

        random.seed(seed)
        np.random.seed(seed)

        super().__init__(**kwargs)

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        self.selectable_tokens = [key for key in self.tokenizer.vocab.keys() if "<" not in key]
        self.shortest_length = shortest_length
        self.longest_length = longest_length
        self.seed = seed

    def __iter__(self):
        """Yields randomly generated peptides forever"""
        while True:
            tokens, length = self.generate_sample()
            tokens = self.tokenizer.add_special_tokens(tokens)
            masked_tokens, mask_targets = self._apply_bert_mask(tokens)
            token_ids = np.array(
                self.tokenizer.convert_tokens_to_ids(tokens), np.int64)
            masked_token_ids = np.array(
                self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
            input_mask = np.ones_like(token_ids)

            yield token_ids, masked_token_ids, mask_targets

    def generate_sample(self, length=50):
        """Returns a random peptide"""
        #length = random.randint(a=self.shortest_length, b=self.longest_length)
        tokens = random.choices(self.selectable_tokens, k=length)
        return tokens

    def create_lmdb(self, directory, n_samples, length_histogram={}, split="train"):
        """Creates an lmdb file with specified number of random samples"""
        # Update with your json data directory path
        if directory[-1] != "/":
            directory += "/"
        result = []
                    
        # Write json objects to LMDB
        print("writing to file")
        map_size = 280000000000 # Taken from Leslie's training map size
        out_dir = Path(directory)
        outfile = 'random_peptide_%s.lmdb' % split
        outpath = str(out_dir/outfile).encode() # Update with the name of your output filename
        print(outpath)
        
        env = lmdb.open(outpath, map_size=map_size, subdir=False, lock=False)
        with env.begin(write=True) as txn:

            for index in range(n_samples):
                if length_histogram:
                    values = [int(key) for key in length_histogram.keys()]
                    total_samples = sum(length_histogram.values())
                    probabilities = [round(value/total_samples,2) for value in length_histogram.values()]
                    sum_prob = round(sum(probabilities),2)
                    if sum_prob != 1.0 and 1.0-sum_prob <= 0.02:
                        #print('I am here')
                        probabilities[0] = probabilities[0]+(1.0-sum_prob)
                    #print('Sum probs:', sum(probabilities))
                    assert(round(sum(probabilities),3) == 1.0)
                    #print(values, probabilities)
                    length = np.random.choice(values,1,p=probabilities, replace=False)[0]
                else:
                    length = random.randint(a=self.shortest_length, b=self.longest_length)
                sample = self.generate_sample(length=length)
                txn.put(str(index).encode(), pkl.dumps(sample))

            txn.put('num_examples'.encode(), pkl.dumps(n_samples)) # Includes this field at end of file following TAPE file format
            
        env.close()

        print("Finished writing")
        print("num lines: ", n_samples)

    def read_lmdb(self, directory):    
        """Reads lmdb to perform sanity check."""
        directory += "random_peptide_train.lmdb"
        filepath = Path(directory)
        env = lmdb.open(str(filepath), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)
        
        #test read num_examples:
        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))
        print("num_examples: ", num_examples)
        
        #test read from index: first 10 entries
        with env.begin(write=False) as txn:
            for index in list(range(0,10)):
                item = pkl.loads(txn.get(str(index).encode()))
                print(index, ' : ', item)

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        """Turns list of samples into batch that can be fed to a model

        Args:
            batch (List[Any]): A list of samples to be turned into a batch.

        Returns:
            A batch corresponding to the given list of samples
        """
        input_ids, masked_input_ids, mask_targets = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        masked_input_ids = torch.from_numpy(pad_sequences(masked_input_ids, 0))
        mask_targets = torch.from_numpy(pad_sequences(mask_targets, 0))
        return {'input_ids': input_ids,
                'masked_input_ids': masked_input_ids,
                'mask_targets': mask_targets}

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
