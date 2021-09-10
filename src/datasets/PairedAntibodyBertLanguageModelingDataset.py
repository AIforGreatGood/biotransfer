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
from .AntibodyBertLanguageModelingDataset import LMDBDatasetForAntibody

class PairedAntibodyBertLanguageModelingDataset(Dataset):
	def __init__(self, split: str, data_path: Union[str, Path], in_memory: bool=False, maxlen=200, **kwargs):
		super().__init__()
		print('args',locals())
		self.tokenizer = TAPETokenizer(vocab='iupac')
		self.start_token = self.tokenizer.start_token
		self.sep_token = self.tokenizer.stop_token
		data_file = f'oas_paired_{split}.lmdb'
		data_path = os.path.join(data_path, data_file)
		data_path = Path(data_path)
		self.data = LMDBDatasetForAntibody(data_path, in_memory)
		self.maxlen = maxlen

	def __len__(self) -> int:
		return len(self.data)

	def __getitem__(self, index):
		item = self.data[index]
		if len(item["seq_heavy"])+len(item["seq_light"])+1 > self.maxlen:
                    extra = self.maxlen-1-len(item["seq_heavy"])-len(item["seq_light"])
                    int_light = int(extra/2)
                    int_heavy = extra-int_light
                    max_heavy = item["seq_heavy"]-int_heavy
                    max_light = item["seq_light"]-int_light
                    item["seq_heavy"] = self.slide_long_sequence(item["seq_heavy"], max_heavy)
                    item["seq_light"] = self.slide_long_sequence(item["seq_light"], max_light)

		tokens_heavy = self.tokenizer.tokenize(item['seq_heavy'])
		tokens_light = self.tokenizer.tokenize(item['seq_light'])

		# add special tokens; 
		# TODO: rewrite TAPE's tokenizer to include adding special tokens for pairs of sequences
		tokens = [self.start_token]+tokens_heavy+[self.sep_token]+tokens_light+[self.sep_token]
		masked_tokens, labels = self._apply_bert_mask(tokens)
		masked_token_ids = np.array(self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
		input_mask = np.ones_like(masked_token_ids)
		token_type_ids = np.array([1]*(len(tokens_heavy)+2) + [2]*(len(tokens_light)+1))
		position_ids = np.array(list(range(1, len(tokens_heavy)+3))+list(range(1, len(tokens_light)+2)))
		return masked_token_ids, input_mask, labels, position_ids, token_type_ids

	def slide_long_sequence(self, seq, max_seqlen):
		start = np.random.randint(len(seq) - max_seqlen + 1)
		end = start + max_seqlen
		seq = seq[start:end]

		return seq

	def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
		input_ids, input_mask, lm_label_ids, lm_position_ids, lm_token_type_ids = tuple(zip(*batch))

		torch_inputs = torch.from_numpy(pad_sequences(input_ids, 0)) 
		input_mask = torch.from_numpy(pad_sequences(input_mask, 0)) 
		torch_labels = torch.from_numpy(pad_sequences(lm_label_ids, -1)) 
		torch_position_ids = torch.from_numpy(pad_sequences(lm_position_ids, 0))
		torch_token_type_ids = torch.from_numpy(pad_sequences(lm_token_type_ids, 0)) # 1 for the first segment, 2 for the second segment and 0 for padding
		return {'input_ids': torch_inputs,
			'input_mask': input_mask,
			'targets': torch_labels,
			'position_ids': torch_position_ids,
			'token_type_ids': torch_token_type_ids}

	def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
		masked_tokens = copy(tokens)
		labels = np.zeros([len(tokens)], np.int64) - 1
		print(tokens)

		flag = 0
		for i, token in enumerate(tokens):
			"""
			# only test the light chain
			if token == self.tokenizer.stop_token:
				flag = 1
				continue
			if flag == 0:
				continue
			"""
			"""
			# only test the heavy chain
			if token == self.tokenizer.stop_token:
				flag = 1
				continue
			if flag == 1:
				continue
			"""

			# Tokens begin and end with start_token and stop_token, ignore these
			if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
				continue

			prob = random.random()
			if prob < 0.15:
				prob /= 0.15
				labels[i] = self.tokenizer.convert_token_to_id(token)

				if prob < 0.8:
					#print('IF <0.8')
					# 80% chance for random change to mask token
					token = self.tokenizer.mask_token
				elif prob < 0.9:
					#print('IF <0.9')
					# 10% chance to change to random token
					token = self.tokenizer.convert_id_to_token(
						random.randint(0, self.tokenizer.vocab_size - 1))
				else:
					#print('ELSE 0.1')
					# 10% chance to keep current token
					pass

				masked_tokens[i] = token
		return masked_tokens, labels