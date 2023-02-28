# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

from copy import copy
import math
from pathlib import Path
import random
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from omegaconf.listconfig import ListConfig

import pandas as pd
import numpy as np
import scipy
from tape.datasets import dataset_factory
from tape.datasets import pad_sequences
from tape.tokenizers import TAPETokenizer
from tape.registry import registry
from tape import ProteinBertForValuePrediction
import torch
from torch.utils.data import Dataset

from collections import Counter
import time

def compute_mutations(seq_0, seq_1):
    assert len(seq_0) == len(seq_1)
    cnt = 0
    for i in range(len(seq_0)):
        if seq_0[i] != seq_1[i]:
            cnt+=1
    return cnt


class PairedCovidDataset(Dataset):
    """
    Dataset pairing antibody backbones with Covid binding affinity measurements.
    # antibodies: 14H, 14L, 95L, 91H

    alpha correction
    Calculated by (1/3)*(sum(assay_1_control) - sum(assay_2_control)), then added against second assay pred affinity.
    """

    # Found in "5 Sequences_Sent_To_AAlpha" word doc that Matt distributed.
    # Includes full static and dynamic portions of protein
    original_full_backbones = {
        "14H": "EVQLVETGGGLVQPGGSLRLSCAASGFTLNSYGISWVRQAPGKGPEWVSVIYSDGRRTFYGDSVKGRFTISRDTSTNTVYLQMNSLRVEDTAVYYCAKGRAAGTFDSWGQGTLVTVSS",
        "14L": "DVVMTQSPESLAVSLGERATISCKSSQSVLYESRNKNSVAWYQQKAGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDAAVYYCQQYHRLPLSFGGGTKVEIK",
        "91H": "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAENSLYLQMNSLRAEDTALYYCAKVGRGGGYFDYWGQGTLVTVSS",
        "91L": "QAVLTQPSSLSASPGASVSLTCTLRSGINVGTYRIYWYQQKPGSPPQYLLRYKSDSDKQQGSGVPSRFSGSKDASANAGILLISGLQSEDEADYYCMIWHSSAWVFGGGTKLTVL",
        "95H": "EVQLVESGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVGRGVIDHWGQGTLVTVSS",
        "95L": "SSELTQDPAVSVALGQTVRITCEGDSLRYYYANWYQQKPGQAPILVIYGKNNRPSGIADRFSGSNSGDTSSLIITGAQAEDEADYYCSSRDSSGFQVFFGAGTKLTVL"
    }

    # Starts at the beginning of the first highlighted segment (CDR1) and ends at the end of the last highlighted segment (CDR3).
    original_truncated_backbones = {
        "14H": "GFTLNSYGISWVRQAPGKGPEWVSVIYSDGRRTFYGDSVKGRFTISRDTSTNTVYLQMNSLRVEDTAVYYCAKGRAAGTFDS",
        "14L": "KSSQSVLYESRNKNSVAWYQQKAGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDAAVYYCQQYHRLPLS",
        "91H": "GFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAENSLYLQMNSLRAEDTALYYCAKVGRGGGYFDY",
        "91L": "TLRSGINVGTYRIYWYQQKPGSPPQYLLRYKSDSDKQQGSGVPSRFSGSKDASANAGILLISGLQSEDEADYYCMIWHSSAWV",
        "95H": "GYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVGRGVIDH",
        "95L": "EGDSLRYYYANWYQQKPGQAPILVIYGKNNRPSGIADRFSGSNSGDTSSLIITGAQAEDEADYYCSSRDSSGFQVF"
    }

    # Indices for the static regions so that they can be removed. Includes the beginnings/ends not present in excel data.
    backbone_static_regions = {
        "14H": [(0, 25), (35, 50), (64, 98), (107, 118)],
        "14L": [(0, 23), (40, 55), (62, 94), (103, 113)],
        "91H": [(0, 25), (35, 49), (65, 98), (108, 119)],
        "91L": [(0, 22), (36, 51), (64, 96), (105, 115)],
        "95H": [(0, 25), (35, 50), (62, 98), (106, 117)],
        "95L": [(0, 22), (33, 48), (55, 87), (98, 108)]
    }


    def __init__(self,
                 chain: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 add_static_ends = True,
                 remove_static = False):
        """
        params
            chain [14H, 14L, 91H, 95L]: The backbone peptide chain to load in. Can be 14H, 14L, 91H, or 95L. Each has around the same amount of data.
            tokenizer: Encodes amino acid characters into ids and appends start/end tokens. Default uses tape's iupac tokenizer.
            add_static_ends: Whether or not to add static ends of peptide to data.
            remove_static: Whether or not to remove static middle region of peptide.
        """
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        self.start_token = self.tokenizer.start_token
        self.sep_token = self.tokenizer.stop_token

        # Assertions: chains
        assert chain in ["14H", "14L", "91H", "95L"]
        self.chain = chain

        # chains
        chain_type = chain[-1]
        if chain_type == 'L':
            fixed_chain_name = chain[:-1]+'H'
        else:
            fixed_chain_name = chain[:-1]+'L'
        fixed_seq = self.__class__.original_truncated_backbones[fixed_chain_name]
        # Adds static regions at the beginning and end from the original backbones to the sequences
        if add_static_ends:
            fixed_seq = self.add_static_ends_transform(fixed_seq, chain=fixed_chain_name)
        # If true, removes any static regions from the sequences. Bad for language modeling, potentially good for VAE.
        if remove_static:
            fixed_seq = self.remove_static_transform(fixed_seq, chain=fixed_chain_name)
        self.fixed_seq = fixed_seq        

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        """Returns sample from dataset at specific index

        Args:
            index (int): Index of dataset

        Returns:
            ???
        """
        item = self.data[index]
        if self.chain[-1]=='L':
            tokens_heavy = self.tokenizer.tokenize(self.fixed_seq)
            tokens_light = self.tokenizer.tokenize(item["aa_seq"])
        elif self.chain[-1]=='H':
            tokens_heavy = self.tokenizer.tokenize(item["aa_seq"])
            tokens_light = self.tokenizer.tokenize(self.fixed_seq)
        
        # add special tokens; 
        # TODO: rewrite TAPE's tokenizer to include adding special tokens for pairs of sequences
        tokens = [self.start_token]+tokens_heavy+[self.sep_token]+tokens_light+[self.sep_token]
        token_ids = np.array(self.tokenizer.convert_tokens_to_ids(tokens), np.int64)
        input_mask = np.ones_like(token_ids)
        token_type_ids = np.array([1]*(len(tokens_heavy)+2) + [2]*(len(tokens_light)+1))
        position_ids = np.array(list(range(1, len(tokens_heavy)+3))+list(range(1, len(tokens_light)+2)))
        if "pred_aff" in item:
            input_label = float(item["pred_aff"])
            return token_ids, input_mask, input_label, position_ids, token_type_ids
        else:
            return token_ids, input_mask, position_ids, token_type_ids
    
    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        """Turns list of samples into batch that can be fed to a model

        Args:
            batch (List[Any]): A list of samples to be turned into a batch.

        Returns:
            A batch corresponding to the given list of samples
        """
        if len(batch[0]) == 5:
            input_ids, input_mask, lm_label_ids, lm_position_ids, lm_token_type_ids = tuple(zip(*batch))
            torch_labels = torch.FloatTensor(lm_label_ids)  # type: ignore
            torch_labels = torch_labels.unsqueeze(1)
        elif len(batch[0]) == 4:
            input_ids, input_mask, lm_position_ids, lm_token_type_ids = tuple(zip(*batch))
        torch_inputs = torch.from_numpy(pad_sequences(input_ids, 0)) 
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0)) 
        torch_position_ids = torch.from_numpy(pad_sequences(lm_position_ids, 0))
        torch_token_type_ids = torch.from_numpy(pad_sequences(lm_token_type_ids, 0)) # 1 for the first segment, 2 for the second segment and 0 for padding
        
        if len(batch[0]) == 5:
            return {'input_ids': torch_inputs,
                    'input_mask': input_mask,
                    'targets': torch_labels,
                    'position_ids': torch_position_ids,
                    'token_type_ids': torch_token_type_ids}
        else:
            return {'input_ids': torch_inputs,
                    'input_mask': input_mask,
                    'position_ids': torch_position_ids,
                    'token_type_ids': torch_token_type_ids}

    def collate_fn_onnx(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        """Turns list of samples into batch that can be fed to an onnx model

        Args:
            batch (List[Any]): A list of samples to be turned into a batch.

        Returns:
            A batch corresponding to the given list of samples
        """
        if len(batch[0]) == 5:
            input_ids, input_mask, true_value, position_ids, token_type_ids = tuple(zip(*batch))
        elif len(batch[0]) == 4:
            input_ids, input_mask, position_ids, token_type_ids = tuple(zip(*batch))

        input_ids = pad_sequences(input_ids, 0)
        input_mask = pad_sequences(input_mask, 0)
        position_ids = pad_sequences(position_ids, 0)
        token_type_ids = pad_sequences(token_type_ids, 0)

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'position_ids': position_ids,
                'token_type_ids': token_type_ids}

    def add_data(self, sequences):
        """Sets data to sequences

        Args:
            sequences: sequences to add
        """
        self.data = [{'aa_seq':s} for s in sequences]
        return

    def add_static_ends_transform(self, x, chain=None):
        """Adds static ends of protein to the beginning and end of the sequences. These static ends are present in the word
           doc and pretraining data, but not in the AAlphaBio excel sheet and pandas dataframe.

        Args:
            x: A sequence of tokens

        Returns:
            The original sequence with static regions added.
        """
        if chain is None:
            chain = self.chain
        static = self.__class__.backbone_static_regions[chain][0]
        add_inds = list(range(static[0], static[1]))
        beginning_static_seq = list(map(self.__class__.original_full_backbones[chain].__getitem__, add_inds))
        try:
            x = "".join(beginning_static_seq) + x
        except:
            print('errrrrrrrrrrrrrrrrrrr', x)

        static = self.__class__.backbone_static_regions[chain][-1]
        add_inds = list(range(static[0], static[1]))
        end_static_seq = list(map(self.__class__.original_full_backbones[chain].__getitem__, add_inds))
        x = x + "".join(end_static_seq)

        return x

    def remove_static_transform(self, x, chain=None):
        """Remove all static portions of the protein. First adds the static 
           ends, then removes all static indices from sample.

        Args:
            x: A sequence of tokens
            
        Returns:
            The original sequence with static regions removed.
        """
        if chain is None:
            chain = self.chain

        x = self.add_static_ends_transform(x)

        remove_inds = []
        for static in self.__class__.backbone_static_regions[chain]:
            inds = list(range(static[0], static[1]))
            for ind in inds:
                remove_inds.append(ind)

        filtered_seq = [x[i] for i in range(len(x)) if (i not in remove_inds)]
        filtered_seq = "".join(filtered_seq)

        return filtered_seq