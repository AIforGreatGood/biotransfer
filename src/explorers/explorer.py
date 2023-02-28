# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

"""Defines abstract base explorer class."""
import abc
import json
import os
import time
import warnings
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tqdm

from src.flexs_modules import Landscape, Model


class Explorer(abc.ABC):
    """
    Abstract base explorer class.
    Run explorer through the `run` method. Implement subclasses
    by overriding `propose_sequences` (do not override `run`).
    """

    def __init__(
        self,
        model: Model,
        name: str,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        log_file: Optional[str] = None,
        ):
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
        self.model = model
        self.name = name

        self.rounds = rounds
        self.sequences_batch_size = sequences_batch_size
        self.model_queries_per_batch = model_queries_per_batch
        self.starting_sequence = starting_sequence

        self.log_file = log_file
        if self.log_file is not None:
            dir_path, filename = os.path.split(self.log_file)
            os.makedirs(dir_path, exist_ok=True)

        if model_queries_per_batch < sequences_batch_size:
            warnings.warn(
                "`model_queries_per_batch` should be >= `sequences_batch_size`"
            )

    @abc.abstractmethod
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
        pass

    def get_starting_sequence(self):
        return self.starting_sequence

    def _log(
        self,
        sequences_data: pd.DataFrame,
        metadata: Dict,
        current_round: int,
        verbose: bool,
        round_start_time: float,
        ) -> None:
        if self.log_file is not None:
            with open(self.log_file, "w") as f:
                # First write metadata
                json.dump(metadata, f)
                f.write("\n")

                # Then write pandas dataframe
                sequences_data.to_csv(f, index=False)

        if verbose:
            if 'true_score' in sequences_data.columns:
                print(
                    f"round: {current_round}, top (true score): {sequences_data['true_score'].max()}, "
                    f"time: {time.time() - round_start_time:02f}s"
                )
            else:
                print(
                    f"round: {current_round}, top (model_score): {sequences_data['model_score'].max()}, "
                    f"time: {time.time() - round_start_time:02f}s"
                )

    def run(self, landscape: Landscape = None, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the exporer.
        Args:
            landscape: Ground truth fitness landscape.
            verbose: Whether to print output or not.
        """
        self.model.cost = 0

        # Metadata about run that will be used for logging purposes
        if landscape:
            metadata = {
                "run_id": datetime.now().strftime("%H:%M:%S-%m/%d/%Y"),
                "explorer_name": self.name,
                "model_name": self.model.name,
                "landscape_name": landscape.name,
                "rounds": self.rounds,
                "sequences_batch_size": self.sequences_batch_size,
                "model_queries_per_batch": self.model_queries_per_batch,
            }
        else:
            metadata = {
                "run_id": datetime.now().strftime("%H:%M:%S-%m/%d/%Y"),
                "explorer_name": self.name,
                "model_name": self.model.name,
                "rounds": self.rounds,
                "sequences_batch_size": self.sequences_batch_size,
                "model_queries_per_batch": self.model_queries_per_batch,
            }

        # Initial sequences and their scores
        model_score, mean, std = self.model.get_fitness([self.starting_sequence])
        print('starting sequence', self.starting_sequence)
        if landscape:
            sequences_data = pd.DataFrame(
                {
                    "sequence": [self.starting_sequence],
                    "model_score": model_score,
                    "mean": mean,
                    "std": std,
                    "true_score": landscape.get_fitness([self.starting_sequence]),
                    "round": [0],
                    "model_cost": [self.model.cost],
                    "measurement_cost": [1],
                }
            )
        else:
            sequences_data = pd.DataFrame(
                {
                    "sequence": [self.starting_sequence],
                    "model_score": model_score,
                    "mean": mean,
                    "std": std,
                    "round": [0],
                    "model_cost": [self.model.cost],
                }
            )

        self._log(sequences_data, metadata, 0, verbose, time.time())

        # For each round, train model on available data, propose sequences,
        # measure them on the true landscape, add to available data, and repeat.
        range_iterator = range if verbose else tqdm.trange
        for r in range_iterator(1, self.rounds + 1):
            round_start_time = time.time()

            if landscape:
                self.model.train( # if the model.train method is defined
                    sequences_data["sequence"].to_numpy(),
                    sequences_data["true_score"].to_numpy(),
                )

            seqs, preds, mean, std = self.propose_sequences(sequences_data)
            if len(preds)==0:
                print('The local maximum is found!')
                return

            if landscape:
                true_score = landscape.get_fitness(seqs)

            if len(seqs) > self.sequences_batch_size:
                warnings.warn(
                    "Must propose <= `self.sequences_batch_size` sequences per round"
                )

            if landscape:
                sequences_data = sequences_data.append(
                    pd.DataFrame(
                        {
                            "sequence": seqs,
                            "model_score": preds,
                            "mean": mean,
                            "std": std,
                            "true_score": true_score,
                            "round": r,
                            "model_cost": self.model.cost,
                            "measurement_cost": len(sequences_data) + len(seqs),
                        }
                    ),
                    ignore_index=True
                )
            else:
                sequences_data = sequences_data.append(
                    pd.DataFrame(
                        {
                            "sequence": seqs,
                            "model_score": preds,
                            "mean": mean,
                            "std": std,
                            "round": r,
                            "model_cost": self.model.cost,
                        }
                    ),
                    ignore_index=True
                )
            self._log(sequences_data, metadata, r, verbose, round_start_time)

        return sequences_data, metadata