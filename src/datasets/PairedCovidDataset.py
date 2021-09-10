# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

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
                 split: str=None,
                 data_path: Union[str, Path] = "/home/gridsan/groups/ai4bio_shared/datasets/AAbio_data/covid/",
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 average_replicates: bool = False,
                 add_static_ends: bool = False,
                 remove_static: bool = False,
                 filter_nan: str = "drop",
                 correction: str = "assay"):
        """
        params
            chain [14H, 14L, 91H, 95L]: The backbone peptide chain to load in. Can be 14H, 14L, 91H, or 95L. Each has around the same amount of data.
            split [train, valid, test]: Train/valid/test split. Percentages are .8/.1/.1, with equal proportions of kmutations and affinity scores.
            data_path: Path to the covid data directory. Default one leads to shared llgrid location.
            tokenizer: Encodes amino acid characters into ids and appends start/end tokens. Default uses tape's iupac tokenizer.
            average_replicates: Every independent peptide has 3 trials of measurements. If False, all measurements are kept as independent
                                 samples. If True, all peptide trials are averaged into one sample.
            add_static_ends: The peptides within the data are missing static sequences on the beginning and end from the original sequence. If True,
                             this adds those static beginnings/ends back to the data. If False, leaves data as is.
            remove_static: The peptides in the data have various dynamic regions which were mutated for experiments, and regions which were kept 
                           completely static during experiments. If True, this removes the static regions, leaving only the dynamic ones. If False,
                           data is kept as is.
            filter_nan [drop, max]: How to handle pred_aff nans in data. Can be dropped or filled with the max value.
            correction [assay, replicate]: Whether to train on the assay corrected data or the replicate corrected data.
                                            Each assay has bias, as well as each replicate.
                                            Assay corrected data calculates a corrective term between the two assays and adds it against the second assay.
                                            Replicate corrected data calculates a corrected term between all assays (2) and replicates (3),
                                            leading to 5 corrective terms that are applied respectively to each assay/replicate combination.
        """
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        self.start_token = self.tokenizer.start_token
        self.sep_token = self.tokenizer.stop_token

        # Assertions: chains
        self.chain = chain

        # Assertion: split
        if split not in ('train', 'valid', 'test', 'all'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test', 'all']")

        # check input
        if add_static_ends and remove_static:
            raise Exception("Cannot add static ends and remove static at the same time. Only one of these options should be True.")

        # Assertion: correction
        assert correction in ["assay", "replicate"]
        if not average_replicates: # correction ='replicate' should be if average_replicates is False
            assert correction == 'replicate'

        # Assertion: filter_nan
        assert filter_nan in ["drop","max","median"]

        # Load dataframe
        self.data_path = data_path
        if self.data_path[-1] != "/":
            self.data_path += "/"
        
        # Compute imputation value
        data = []
        if filter_nan in ['max', 'median']:
            c = chain
            data_file = self.data_path + '{}_corrected/{}/{}_{}.csv'.format(correction, c, c, 'train')
            data.append(pd.read_csv(data_file))
            data_file = self.data_path + '{}_corrected/{}/{}_{}.csv'.format(correction, c, c, 'valid')
            data.append(pd.read_csv(data_file))
            data_file = self.data_path + '{}_corrected/{}/{}_{}.csv'.format(correction, c, c, 'test')
            data.append(pd.read_csv(data_file))
            data = pd.concat(data) 
            if filter_nan == 'max':
                imputation_val = data["pred_aff"].max()
            else:
                data = data.dropna()
                grouped_data = data.groupby(['mata_description'])
                grouped_data = grouped_data.filter(lambda x: len(x) < 3)
                imputation_val = grouped_data["pred_aff"].median()

        # Load data with the pre-defined split
        self.fixed_seq = {}
        c = chain
        data_file = self.data_path + '{}_corrected/{}/{}_{}.csv'.format(correction, c, c, split)
        data_df = pd.read_csv(data_file)
        # chains
        assert c[-1] in ['H', 'L']
        chain_type = c[-1]
        if chain_type == 'L':
            fixed_chain_name = c[:-1]+'H'
        else:
            fixed_chain_name = c[:-1]+'L'
        fixed_seq = self.__class__.original_truncated_backbones[fixed_chain_name]
        # Adds static regions at the beginning and end from the original backbones to the sequences
        if add_static_ends:
            data_df["aa_seq"] = data_df.aa_seq.transform(self.add_static_ends_transform,chain=c)
            fixed_seq = self.add_static_ends_transform(fixed_seq, chain=fixed_chain_name)
        # If true, removes any static regions from the sequences. Bad for language modeling, potentially good for VAE.
        if remove_static:
            data_df["aa_seq"] = data_df.aa_seq.transform(self.remove_static_transform,chain=c)
            fixed_seq = self.remove_static_transform(fixed_seq, chain=fixed_chain_name)
        self.fixed_seq=fixed_seq
        data = data_df           

        # If "max", replace nans with max affinity values
        # If "drop", drop nans
        if split == 'test': # only use data that have at least one measurement
            data = data.groupby(['mata_description']).filter(lambda x: len(x.dropna())>0)
        if filter_nan == "drop":
            data = data.dropna()
        else:
            data = data.fillna(imputation_val)

        # If true, average replicates over the same sequence and drop extra replicates
        if average_replicates:
            data['pred_aff'] = data.groupby(['mata_description']).pred_aff.transform('mean')
            data = data.drop_duplicates(subset=['mata_description'])

        # Only use sequence and label
        self.data = data.to_dict('records')

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
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

    def collate_fn_numpy(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        if len(batch[0]) == 5:
            input_ids, input_mask, lm_label_ids, lm_position_ids, lm_token_type_ids = tuple(zip(*batch))
        elif len(batch[0]) == 4:
            input_ids, input_mask, lm_position_ids, lm_token_type_ids = tuple(zip(*batch))
        torch_inputs = torch.from_numpy(pad_sequences(input_ids, 0)) 
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0)) 
        return {'input_ids': torch_inputs,
                'input_mask': input_mask}

    def add_static_ends_transform(self, x, chain=None):
        """
        Adds static ends of protein to the beginning and end of the sequences. These static ends are present in the word
        doc and pretraining data, but not in the AAlphaBio excel sheet and pandas dataframe.
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
        """
        Remove all static portions of the protein. First adds the static ends, then removes all static indices from sample.
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