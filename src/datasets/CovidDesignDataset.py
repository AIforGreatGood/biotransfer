# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

from copy import copy
import math
from pathlib import Path
import random
import os
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection

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
import itertools


class CovidDataset(Dataset):
    """
    Dataset pairing antibody backbones with Covid binding affinity measurements.

    Alpha correction calculated by (1/3)*(sum(assay_1_control) - sum(assay_2_control)),
    then added against second assay pred affinity.
    """

    # Found in "5 Sequences_Sent_To_AAlpha" word doc that Matt distributed.
    # Includes full static and dynamic portions of protein
    original_full_backbones = {
        "14H": "EVQLVETGGGLVQPGGSLRLSCAASGFTLNSYGISWVRQAPGKGPEWVSVIYSDGRRTFYGDSVKGRFTISRDTSTNTVYLQMNSLRVEDTAVYYCAKGRAAGTFDSWGQGTLVTVSS",
        "14L": "DVVMTQSPESLAVSLGERATISCKSSQSVLYESRNKNSVAWYQQKAGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDAAVYYCQQYHRLPLSFGGGTKVEIK",
        "91H": "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAENSLYLQMNSLRAEDTALYYCAKVGRGGGYFDYWGQGTLVTVSS",
        "95L": "SSELTQDPAVSVALGQTVRITCEGDSLRYYYANWYQQKPGQAPILVIYGKNNRPSGIADRFSGSNSGDTSSLIITGAQAEDEADYYCSSRDSSGFQVFFGAGTKLTVL"
    }

    best_full_backbones = {
        "14H": "EVQLVETGGGLVQPGGSLRLSCAASGFTLNEYGISWVRQAPGKGPEWVSVIYSDGRRTFYGDSVKGRFTISRDTSTNTVYLQMNSLRVEDTAVYYCAKGRAAVTFDQWGQGTLVTVSS", #0.204156
        "14L": "DVVMTQSPESLAVSLGERATISCHSSPSVLYESRNKNSVAWYQQKAGQPPKLLIYNASTRESGVPDRFSGSGSGTDFTLTISSLQAEDAAVYYCQQYHRLPLSFGGGTKVEIK", #-0.192363
        "91H": "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYGMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAENSLYLQMNSLRAEDTALYYCAKVQRGRGYFDYWGQGTLVTVSS",#-0.736321
        "95L": "SSELTQDPAVSVALGQTVRITCEGDSLRDQYANWYQQKPGQAPILVIYGKNNRPSGIADRFSGSNSGDTSSLIITGAQAEDEADYYCSSRDSSGFQVFFGAGTKLTVL" # -0.0129877
    }

    # Starts at the beginning of the first highlighted segment (CDR1) and ends at the end of the last highlighted segment (CDR3).
    original_truncated_backbones = {
        "14H": "GFTLNSYGISWVRQAPGKGPEWVSVIYSDGRRTFYGDSVKGRFTISRDTSTNTVYLQMNSLRVEDTAVYYCAKGRAAGTFDS",
        "14L": "KSSQSVLYESRNKNSVAWYQQKAGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDAAVYYCQQYHRLPLS",
        "91H": "GFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAENSLYLQMNSLRAEDTALYYCAKVGRGGGYFDY",
        "95L": "EGDSLRYYYANWYQQKPGQAPILVIYGKNNRPSGIADRFSGSNSGDTSSLIITGAQAEDEADYYCSSRDSSGFQVF"
    }

    # Indices for the static regions so that they can be removed. Includes the beginnings/ends not present in excel data.
    backbone_static_regions = {
        "14H": [(0, 25), (35, 50), (64, 98), (107, 118)],
        "14L": [(0, 23), (40, 55), (62, 94), (103, 113)],
        "91H": [(0, 25), (35, 49), (65, 98), (108, 119)],
        "95L": [(0, 22), (33, 48), (55, 87), (98, 108)]
    }

    backbone_variable_regions = {
        "14H": [(25, 35), (50, 64), (98, 107)],
        "14L": [(23, 40), (55, 62), (94, 103)],
        "91H": [(25, 35), (49, 65), (98, 108)],
        "95L": [(22, 33), (48, 55), (87, 98)]
    }

    seed_sequence_value = {
        "14H": 1.3445717628028468,
        "14L": 1.3445717628028468, #mu=1.0022633, std=0.17974249 (predicted)
        "91H": 1.88361976098417,
        "95L": 2.3378946100102533,
    }


    def __init__(self,
                 chain: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 data_path = None):
        """
        Args:
            chain [14H, 14L, 91H, 95L]: The backbone peptide chain to load in. Can be 14H, 14L, 91H, or 95L. Each has around the same amount of data.
            tokenizer: Encodes amino acid characters into ids and appends start/end tokens. Default uses tape's iupac tokenizer.
            data_path: Path to the covid data directory. Default one leads to shared llgrid location.
        """
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        assert chain in ["14H", "14L", "91H", "95L"]
        self.chain = chain

        if data_path is not None:
            self.load_all_data(data_path)

    
    def get_seed_sequence(self,k:list=0):
        print('starting sequence topK:', k)
        topK = self.get_topK_sequences(topk=max(10,k))
        return topK[k][0], self.__class__.seed_sequence_value[self.chain]

    def get_variable_regions(self, cdr_region:int=None):
        """Returns list of indices representing the variable regions of the
           dataset chain type.
        """
        variable_regions = self.__class__.backbone_variable_regions[self.chain]
        if cdr_region:
            assert cdr_region in [1,2,3], 'cdr_region can only be None, 1, 2 or 3'
            r = variable_regions[cdr_region-1]
            variable_token_ind = list(range(r[0], r[1]))
        else: # all cdr regions
            variable_token_ind = [list(range(r[0], r[1])) for r in variable_regions]
            variable_token_ind = list(itertools.chain(*variable_token_ind))
        assert len(variable_token_ind) == len(set(variable_token_ind))
        return variable_token_ind

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        """Returns sample from dataset at specific index

        Args:
            index (int): Index of dataset

        Returns:
            The token ids and input mask of index
        """
        #item = self.data.iloc[index][["aa_seq", "pred_aff"]]
        item = self.data[index]
        tokens = self.tokenizer.tokenize(item["aa_seq"])
        tokens = self.tokenizer.add_special_tokens(tokens)
        token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(tokens), np.int64)
        input_mask = np.ones_like(token_ids)
        if "pred_aff" in item:
            input_label = float(item["pred_aff"]) #np.sign(float(item["pred_aff"]))*np.power(abs(float(item["pred_aff"])),0.5)
            return token_ids, input_mask, input_label
        else:
            return token_ids, input_mask

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        """Turns list of samples into batch that can be fed to a model

        Args:
            batch (List[Any]): A list of samples to be turned into a batch.

        Returns:
            A batch corresponding to the given list of samples
        """
        if len(batch[0]) == 3:
            input_ids, input_mask, true_value = tuple(zip(*batch))
            true_value = torch.FloatTensor(true_value)  # type: ignore
            true_value = true_value.unsqueeze(1)
        elif len(batch[0]) == 2:
            input_ids, input_mask = tuple(zip(*batch))

        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))

        if len(batch[0]) == 3:
            return {'input_ids': input_ids,
                    'input_mask': input_mask,
                    'targets': true_value}
        else:
            return {'input_ids': input_ids,
                    'input_mask': input_mask}

    def collate_fn_onnx(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        """Turns list of samples into batch that can be fed to an onnx model

        Args:
            batch (List[Any]): A list of samples to be turned into a batch.

        Returns:
            A batch corresponding to the given list of samples
        """
        if len(batch[0]) == 3:
            input_ids, input_mask, true_value = tuple(zip(*batch))
            true_value = torch.FloatTensor(true_value)  # type: ignore
            true_value = true_value.unsqueeze(1)
        elif len(batch[0]) == 2:
            input_ids, input_mask = tuple(zip(*batch))

        input_ids = pad_sequences(input_ids, 0)
        input_mask = pad_sequences(input_mask, 0)

        if len(batch[0]) == 3:
            return {'input_ids': input_ids,
                    'input_mask': input_mask,
                    'targets':true_value}
        else:
            return {'input_ids': input_ids,
                    'input_mask': input_mask}

    def add_data(self, sequences):
        """Sets data to sequences

        Args:
            sequences: sequences to add
        """
        self.data = [{'aa_seq':s} for s in sequences]
        return

    def load_all_data(self, data_path, correction='replicate', add_static_ends=True, remove_static=False):
        data_file = data_path + '{}_corrected/{}/{}_{}.csv'.format(correction, self.chain, self.chain, 'train')
        data_ = [pd.read_csv(data_file)]
        #data_file = data_path + '{}_corrected/{}/{}_{}.csv'.format(correction, self.chain, self.chain, 'valid')
        #data_.append(pd.read_csv(data_file))
        #data_file = data_path + '{}_corrected/{}/{}_{}.csv'.format(correction, self.chain, self.chain, 'test')
        #data_.append(pd.read_csv(data_file))
        data = pd.concat(data_) 
        # If true, average replicates over the same sequence and drop extra replicates
        data = data.dropna()
        data['pred_aff'] = data.groupby(['mata_description']).pred_aff.transform('mean')
        data = data.drop_duplicates(subset=['mata_description'])
        # If true, adds static regions at the beginning and end from the original backbones to the sequences
        if add_static_ends:
            data_orig = data.copy(deep=True)
            data["aa_seq"] = data.aa_seq.transform(self.add_static_ends_transform)
        # If true, removes any static regions from the sequences. Bad for language modeling, potentially good for VAE.
        if remove_static:
            data["aa_seq"] = data.aa_seq.transform(self.remove_static_transform)
        self.data = data.to_dict('records')

    def get_topK_sequences(self, topk=10):
        # sort by pred_aff value
        data = pd.DataFrame(self.data)
        data = data.nsmallest(topk, 'pred_aff')
        data_topK = list(zip(data["aa_seq"], data["pred_aff"]))
        print(data_topK)
        return data_topK
        

    def add_static_ends_transform(self, x):
        """Adds static ends of protein to the beginning and end of the sequences. These static ends are present in the word
           doc and pretraining data, but not in the AAlphaBio excel sheet and pandas dataframe.

        Args:
            x: A sequence of tokens

        Returns:
            The original sequence with static regions added.
        """
        add_inds = []
        static = self.__class__.backbone_static_regions[self.chain][0]
        inds = list(range(static[0], static[1]))
        for ind in inds:
            add_inds.append(ind)
        beginning_static_seq = list(map(self.__class__.original_full_backbones[self.chain].__getitem__, add_inds))
        x = "".join(beginning_static_seq) + x


        add_inds = []
        static = self.__class__.backbone_static_regions[self.chain][-1]
        inds = list(range(static[0], static[1]))
        for ind in inds:
            add_inds.append(ind)
        end_static_seq = list(map(self.__class__.original_full_backbones[self.chain].__getitem__, add_inds))
        x = x + "".join(end_static_seq)

        return x

    def remove_static_transform(self, x):
        """Remove all static portions of the protein. First adds the static 
           ends, then removes all static indices from sample.

        Args:
            x: A sequence of tokens
            
        Returns:
            The original sequence with static regions removed.
        """

        x = self.add_static_ends_transform(x)

        remove_inds = []
        for static in self.__class__.backbone_static_regions[self.chain]:
            inds = list(range(static[0], static[1]))
            for ind in inds:
                remove_inds.append(ind)

        filtered_seq = [x[i] for i in range(len(x)) if (i not in remove_inds)]
        filtered_seq = "".join(filtered_seq)

        return filtered_seq