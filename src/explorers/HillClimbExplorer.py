# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

"""Define a hill climb algorithm implementation."""

import random
from time import time
from typing import Callable
from typing import Optional, Tuple

from src.flexs_modules import Model
from .utils import generate_random_mutant, generate_hillclimb_mutations
import numpy as np
import pandas as pd
from .explorer import Explorer
            

class HillClimbExplorer(Explorer):
    """
    Create an Explorer.

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
        random_restart: bool=True,
        expected_num_mutantations_random_restart: int=2,
        verbose: bool=True,
        log_file: Optional[str] = None,
        ):

        name = (
            "HillClimb"
        )
        if random_restart:
            print('random restart')
            starting_sequence = generate_random_mutant(starting_sequence, expected_num_mutantations=expected_num_mutantations_random_restart, alphabet=alphabet, variable_regions=variable_regions)

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
        self.verbose = verbose

    def propose_sequences(self, measured_sequences_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Propose a list of sequences to be measured in the next round.
        This method will be overriden to contain the explorer logic for each explorer.

        Args:
            measured_sequences_data: A pandas dataframe of all sequences that have been
              measured by the ground truth so far. Has columns "sequence",
              "true_score", "model_score", and "round".

        Returns:
            A tuple containing the proposed sequences and their scores
                (according to the model).
        """
        #print('measured_sequences_data', measured_sequences_data)
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

        next_seq, next_val, mean, std = self.find_next(best_seq, best_val)
        return next_seq, next_val, mean, std


    def find_next(self,curr_seq, curr_val):
        """Finds sequences with better predicted value than the current sequence

        Args:
            curr_seq: Current sequence
            curr_val: Current value for sequence
        """
        targets = np.array(generate_hillclimb_mutations(curr_seq, num_mutations=self.model_queries_per_batch, alphabet=self.alphabet, variable_regions=self.variable_regions))
        target_vals, mean, std = self.model.get_fitness(targets)
        ind = np.where(target_vals > curr_val)[0]
        targets = targets[ind]
        target_vals = target_vals[ind]
        mean = mean[ind]
        std = std[ind]
        # rank 
        sorted_order = np.argsort(target_vals)[: -self.sequences_batch_size : -1]
        targets = targets[sorted_order]
        target_vals = target_vals[sorted_order]
        mean = mean[sorted_order]
        std = std[sorted_order]
        return targets, target_vals, mean, std







    

