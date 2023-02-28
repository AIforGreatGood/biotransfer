# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

"""Define a Gibbs sampling implementation."""

from src.flexs_modules import Model
import numpy as np
import pandas as pd
import random
import tqdm
import time
from typing import Callable, List, Optional, Tuple

from .explorer import Explorer
from .utils import generate_k1_mutations

class GibbsSamplingExplorer(Explorer):
    """
    Create an Explorer.
    
    Gibbs sampling samples conditionally on a previous sample at a random index.
    We start with the seed sequence. We choose one index to mutate. We generate
    all mutations from that position. We create a probability distribution over
    all mutants based on their fitness. We sample from this distribution to
    select the next sequence. The chosen sequence seeds the next Gibbs
    sampling round. This repeats for a given number of rounds.

    Args:
        model: Model of ground truth that the explorer will use to help guide
                sequence proposal.
        name: A human-readable name for the explorer (may include parameter values).
        rounds: Number of rounds to run for (a round consists of sequence proposal,
                ground truth fitness measurement of proposed sequences, and retraining
                the model).
        sequences_batch_size: Number of sequences to propose for measurement from
                ground truth per round.
        model_queries_per_batch: Number of allowed "in-silico" model evaluations
                per round.
        starting_sequence: Sequence from which to start exploration.
        log_file: .csv filepath to write output.
    """
    def __init__(
        self,
        model: Model,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        alphabet: str,
        variable_regions: list=None,
        gamma: float=3, 
        verbose: bool=True,
        log_file: Optional[str] = None,
        ):

        name = (
            "GibbsSampling"
        )

        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )

        self.alphabet = alphabet
        self.variable_regions = variable_regions
        self.gamma = gamma
        self.verbose = verbose

    def propose_sequences(self, measured_sequences_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propose a list of sequences to be measured in the next round.
        This method will be overriden to contain the explorer logic for each explorer.
        Args:
            measured_sequences_data: A pandas dataframe of all sequences that have been
            measured by the ground truth so far. Has columns "sequence",
            "true_score", "model_score", and "round".
        Returns:
            A tuple containing the proposed sequences and their scores
                (according to the model).
        """
        print('measured_sequences_data', measured_sequences_data)
        if 'true_score' in measured_sequences_data.columns:
            best_index = measured_sequences_data['true_score'].idxmax()
            best_val = measured_sequences_data.iloc[best_index]["true_score"]
        else:
            best_index = measured_sequences_data['model_score'].idxmax()
            best_val = measured_sequences_data.iloc[best_index]["model_score"]
        best_seq = measured_sequences_data.iloc[best_index]["sequence"]
        print('best val:',best_val) 
        print('predicted mean:', measured_sequences_data.iloc[best_index]["mean"])
        print('predicted std:', measured_sequences_data.iloc[best_index]["std"])

        last_record = measured_sequences_data.iloc[-1]
        next_seq, next_val, mean, std = self.gibbs_sampling(last_record["sequence"])
        return next_seq, next_val, mean, std

    def gibbs_sampling(self, seq):
        """ Randomly sample one token at a time based on given sequence """
        sequences = []
        sequence_vals = []
        means = []
        stds = []
        for iter in tqdm.tqdm(range(self.sequences_batch_size)):
            index = random.choice(self.variable_regions) # randomly pick an index from the variable regions
            targets = generate_k1_mutations(seq, index, self.alphabet)
            target_log_cdf, mean, std = self.model.get_fitness(targets) 
            sample_dist = np.exp(self.gamma*target_log_cdf)/sum(np.exp(self.gamma*target_log_cdf))
            print('target_log_cdf:', target_log_cdf)
            print('sample_dist:', sample_dist)
            sampled_ind = np.random.choice(list(range(len(target_log_cdf))),p=sample_dist) 
            seq = targets[sampled_ind]
            sequences.append(seq)
            sequence_vals.append(target_log_cdf[sampled_ind])
            means.append(mean[sampled_ind])
            stds.append(std[sampled_ind])
        return sequences, sequence_vals, means, stds