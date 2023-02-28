# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

from collections import OrderedDict
import os
import socket

import numpy as np

###

def parse_node(node):
    """Parse a slurm node name

    Args:
        node: node name

    Returns:
        Parsed node name
    """
    prefix = node[:node.find("[")]
    node_range = node[node.find("[")+1:node.find("]")]
    split_range = node_range.split("-")
    split_range = [int(i) for i in split_range]
    split_range = [str(i) for i in range(split_range[0], split_range[1]+1)]

    parsed_node = [prefix+i for i in split_range]
    return parsed_node


def parse_nodelist(nodelist):
    """Parse a slurm nodelist

    Args:
        nodelist: List of node names

    Returns
        Parsed list of node names
    """
    split_nodelist = nodelist.split(",")

    parsed_nodelist = []
    for node in split_nodelist:

        if "[" in node and "]" in node:
            node = parse_node(node)
        
        if type(node) is list:
            parsed_nodelist = parsed_nodelist + node
        else:
            parsed_nodelist.append(node)

    return parsed_nodelist


def parse_env4lightning(verbose=False, nccl_debug=None):
    """
    The following code is mostly designed to work around
    some issues with Lightning and non-standard slurm configurations

    Modified from Raiden

    Args:
        verbose: Whether or not to print extra statements
        nccl_debug: Whether or not to print nccl debug statements
    """
    if os.environ.get("PARSE_4_LIGHTNING", "0") == "1":
        return

    os.environ["HYDRA_FULL_ERROR"] = "1"
    if nccl_debug is not None:
        os.environ["NCCL_DEBUG"] = nccl_debug

    # fix GPU device ids.
    # Reason: CUDA is labeled with gpu names rather than ordinals
    if 'CUDA_VISIBLE_DEVICES' in os.environ:

        if verbose:
            print("CUDA_VISIBLE_DEVICES before processing: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

        num_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',').__len__()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(f'{i}' for i in range(num_gpus))

        if verbose:
            print("CUDA_VISIBLE_DEVICES after processing: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    elif verbose:
        print("CUDA_VISIBLE_DEVICES not found.")

    # Fix nodelist
    # Reason: llgrid combines consecutive hostnames into 1
    if "SLURM_NODELIST" in os.environ:
        named_nodelist = parse_nodelist(os.environ["SLURM_NODELIST"])
        os.environ["SLURM_NODELIST"] = " ".join(named_nodelist)

        if verbose:
            print("SLURM_NODELIST: {}".format(os.environ["SLURM_NODELIST"]))
    elif verbose:
        print("SLURM_NODELIST not found.")

    if verbose:
        print("HOSTNAME: {}".format(socket.gethostname()))

    os.environ["MASTER_ADDR"] = os.environ["SLURM_NODELIST"].split(" ")[0]
    os.environ["MASTER_PORT"] = "30480"

    # Set flag so this won't be repeated
    os.environ["PARSE_4_LIGHTNING"] = "1"


class Uniprot21Tokenizer:
    r"""TAPE Tokenizer. Can use different vocabs depending on the model.
    """

    def __init__(self):
        self.vocab = OrderedDict([
                                ("A", 0),
                                ("R", 1),
                                ("N", 2),
                                ("D", 3),
                                ("C", 4),
                                ("Q", 5),
                                ("E", 6),
                                ("G", 7),
                                ("H", 8),
                                ("I", 9),
                                ("L", 10),
                                ("K", 11),
                                ("M", 12),
                                ("F", 13),
                                ("P", 14),
                                ("S", 15),
                                ("T", 16),
                                ("W", 17),
                                ("Y", 18),
                                ("V", 19),
                                ("X", 20),
                                ("O", 11),
                                ("U", 4),
                                ("B", 20),
                                ("Z", 20)])
        self.tokens = list(self.vocab.keys())
        self._vocab_type = "uniprot21"
        #assert self.start_token in self.vocab and self.stop_token in self.vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def start_token(self):
        return "<cls>"

    @property
    def stop_token(self):
        return "<sep>"

    @property
    def mask_token(self):
        if "<mask>" in self.vocab:
            return "<mask>"
        else:
            raise RuntimeError(f"{self._vocab_type} vocab does not support masking")

    def tokenize(self, text):
        return [x for x in text]

    def convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        try:
            return self.vocab[token]
        except KeyError:
            raise KeyError(f"Unrecognized token: '{token}'")

    def convert_tokens_to_ids(self, tokens):
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        try:
            return self.tokens[index]
        except IndexError:
            raise IndexError(f"Unrecognized index: '{index}'")

    def convert_ids_to_tokens(self, indices):
        return [self.convert_id_to_token(id_) for id_ in indices]

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        return ''.join(tokens)

    def add_special_tokens(self, token_ids):
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """
        cls_token = [self.start_token]
        sep_token = [self.stop_token]
        return cls_token + token_ids + sep_token

    def encode(self, text):
        tokens = self.tokenize(text)
        #tokens = self.add_special_tokens(tokens)
        token_ids = self.convert_tokens_to_ids(tokens)
        return np.array(token_ids, np.int64)

    @classmethod
    def from_pretrained(cls, **kwargs):
        return cls()