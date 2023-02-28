# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

"""Define a genetic algorithm implementation."""

import random
from typing import Optional, Tuple
from src.flexs_modules import Model
from .utils import generate_random_mutant
from .explorer import Explorer
import numpy as np
import pandas as pd
import torch

class GeneticAlgorithmExplorer(Explorer):
    """A genetic algorithm explorer with single point mutations and recombination.
    Based on the `parent_selection_strategy`, this class implements one of three
    genetic algorithms:

    1. If `parent_selection_strategy == 'top-k'`, we have a traditional
       genetic algorithm where the top-k scoring sequences in the
       population become parents.
    2. If `parent_selection_strategy == 'wright-fisher'`, we have a
       genetic algorithm based off of the Wright-Fisher model of evolution,
       where members of the population become parents with a probability
       exponential to their fitness (softmax the scores then sample).
    """

    def __init__(
        self,
        model: Model,
        rounds: int,
        starting_sequence: str,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        alphabet: str,
        population_size: int,
        parent_selection_strategy: str,
        children_proportion: float,
        parent_selection_proportion: Optional[float] = None,
        beta: Optional[float] = None,
        variable_regions: list = None,
        seed: Optional[int] = None,
        log_file: Optional[str] = None,
    ):
        """Create genetic algorithm.

        Args:
            model: Flexs model
            rounds: Number of design rounds
            start_sequence: Sequence to start designing from
            sequences_batch_size: Batch size of sequences
            model_queries_per_batch: How many times to query the model per batch
            alphabet: What alphabet to use for protein generation
            population_size: Size of the population
            parent_selection_strategy: How to select the parents
            children_proportion: Proportion of children to population
            parent_selection_proportion: Proportion of parents to population
            beta: Exponential coefficient
            variable_regions: The list of possible variable regions
            random_restart: Whether or not to random restart
            seed: Random seed
            log_file: File to log to
        """
        name = (
            f"GeneticAlgorithm_pop_size={population_size}_"
            f"parents={parent_selection_strategy}"
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

        self.variable_regions = variable_regions
        self.alphabet = alphabet
        self.population_size = population_size

        # Validate parent_selection_strategy
        valid_parent_selection_strategies = ["top-proportion", "wright-fisher"]
        if parent_selection_strategy not in valid_parent_selection_strategies:
            raise ValueError(
                f"parent_selection_strategy must be one of "
                f"{valid_parent_selection_strategies}"
            )
        if (
            parent_selection_strategy == "top-proportion"
            and parent_selection_proportion is None
        ):
            raise ValueError(
                "if top-proportion, parent_selection_proportion cannot be None"
            )
        if parent_selection_strategy == "wright-fisher" and beta is None:
            raise ValueError("if wright-fisher, beta cannot be None")
        self.parent_selection_strategy = parent_selection_strategy
        self.beta = beta

        self.children_proportion = children_proportion
        self.parent_selection_proportion = parent_selection_proportion

        self.rng = np.random.default_rng(seed)

        num_children = int(self.children_proportion * self.population_size)
        if num_children % 2 != 0:
            raise Exception(
                "Number of children {} needs to be even to work with recombination strategy. Change population size or children proportion.".format(
                num_children))

    def _choose_parents(self, scores, num_parents):
        """Return parent indices according to `self.parent_selection_strategy`.

        Args:
            scores: Scores from population
            num_parents: Number of parents from population

        Returns
            Distribution over scores
        """
        if self.parent_selection_strategy == "top-proportion":
            k = int(self.parent_selection_proportion * self.population_size)
            return self.rng.choice(np.argsort(scores)[-k:], num_parents)

        # Then self.parent_selection_strategy == "wright-fisher":
        fitnesses = np.exp(scores/self.beta)
        assert np.sum(fitnesses) != 0, 'The model score is too small. Adjust the beta value.'
        probs = fitnesses / np.sum(fitnesses)
        dist = np.random.choice(list(range(len(probs))),size=num_parents, p=probs,replace=True)
        return dist

    def recombination_strategy(self, parents, measured_sequence_set, sequences):
        """Perform single-point crossover on two parents. Parents are already selected and shuffled, so take first two inds
           without replacement.

        Each crossover has two children: one with the parent1/parent2 genes, and one with parent2/parent1 genes

        Args:
            parents: Parents to crossover
            measured_sequence_set: Set of already measured sequences
            sequences: Current sequences in use

        Returns
            Children that are product of crossover
        """
        children_before_mutation = []
        for i in range(int(len(parents)/2)):
            current_parents = parents[2*i:2*i+2]

            crossover_index = random.randint(1, len(parents[0])-1)
            parent1_first = current_parents[0][:crossover_index]
            parent1_second = current_parents[0][crossover_index:]
            parent2_first = current_parents[1][:crossover_index]
            parent2_second = current_parents[1][crossover_index:]

            child1 = "".join([parent1_first, parent2_second])
            child2 = "".join([parent2_first, parent1_second])

            children_before_mutation.append(child1)
            children_before_mutation.append(child2)

        # Single-point mutation of children (expected number of mutated token is 1)
        children = []
        for seq in children_before_mutation:
            child = generate_random_mutant(seq, expected_num_mutantations=1, alphabet=self.alphabet, variable_regions=self.variable_regions) 

            if child not in measured_sequence_set and child not in sequences:
                children.append(child)

        if len(children) == 0:
            return children

        children = np.array(children)
        return children

    def propose_sequences(self, measured_sequences: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation.

        Args:
            measured_sequences: Already measured sequences

        Returns:
            New sequences for population and predictions for those new sequences
        """
        # Set the torch seed by generating a random integer from the pre-seeded self.rng
        torch.manual_seed(self.rng.integers(-(2 ** 31), 2 ** 31))
        
        # Create initial population by choosing parents from `measured_sequences`
        if 'true_score' in measured_sequences.columns:
            initial_pop_inds = self._choose_parents(
                measured_sequences["true_score"].to_numpy(),
                self.population_size,
            )
            scores = measured_sequences["true_score"].to_numpy()[initial_pop_inds]
        else:
            initial_pop_inds = self._choose_parents(
                measured_sequences["model_score"].to_numpy(),
                self.population_size,
            )   
            scores = measured_sequences["model_score"].to_numpy()[initial_pop_inds]        
        pop = measured_sequences["sequence"].to_numpy()[initial_pop_inds] # parent population

        measured_sequence_set = set(measured_sequences["sequence"])
        sequences = {}
        initial_cost = self.model.cost
        while (self.model.cost - initial_cost + self.population_size< self.model_queries_per_batch):
            # Create "children" by recombining parents selected from population
            # according to self.parent_selection_strategy and
            # self.recombination_strategy
            num_children = int(self.children_proportion * self.population_size)
            parents = pop[self._choose_parents(scores, num_children)]
            random.shuffle(parents)

            children = self.recombination_strategy(parents, measured_sequence_set, sequences)
            if len(children) == 0:
                continue

            child_scores, mean, std = self.model.get_fitness(children)

            # Now kick out the worst samples and replace them with the new children
            argsorted_scores = np.argsort(scores)
            pop[argsorted_scores[: len(children)]] = children
            scores[argsorted_scores[: len(children)]] = child_scores

            sequences.update(zip(children, list(zip(child_scores,mean,std))))

        # We propose the top `self.sequences_batch_size`
        # new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        preds, mean, std = zip(*list(sequences.values()))
        preds = np.array(preds)
        mean = np.array(mean)
        std = np.array(std)
        sorted_order = np.argsort(preds)[: -self.sequences_batch_size : -1]

        seqs_sorted = new_seqs[sorted_order]
        preds_sorted = preds[sorted_order]
        mean_sorted = mean[sorted_order]
        std_sorted = std[sorted_order]
        print('best val:', preds_sorted[0], 'mean:',mean_sorted[0], 'std:', std_sorted[0])
        return new_seqs[sorted_order], preds_sorted, mean_sorted, std_sorted