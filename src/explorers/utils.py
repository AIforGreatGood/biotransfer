# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

import random
import itertools

def generate_k1_mutations(seq: str, ind: int, alphabet: str): 
    """
    Generate all k=1 mutations at a specified position
    """
    targets = []
    targets += [seq[:ind]+letter+seq[ind+1:] for letter in alphabet]
    return targets  

def generate_random_mutant(sequence: str, expected_num_mutantations: int, alphabet: str, variable_regions: list=None) -> str:
    """
    Generate a mutant of `sequence` where each residue mutates with probability `mu`.
    So the expected value of the total number of mutations is `len(sequence) * mu`.

    Args:
        sequence: Sequence that will be mutated from.
        expected_num_mutantations: expected number of total mutations
        alphabet: Alphabet string.
        variable_regions: list of integer tuples describing the positions that can be mutations. If None, all positions in the sequence can be mutated.
        
    Returns:
        Mutant sequence string.
    """
    if variable_regions:
        variable_token_ind = variable_regions
    else:
        variable_token_ind = list(range(len(sequences)))
    mu = expected_num_mutantations/len(variable_token_ind) # Probability of mutation per residue.
    
    mutant = []
    for ind, s in enumerate(sequence):
        if ind in variable_token_ind:
            if random.random() < mu:
                mutant.append(random.choice(alphabet))
            else:
                mutant.append(s)
        else:
            mutant.append(s)
    mutant = "".join(mutant)
    assert len(mutant) == len(sequence)
    return mutant

def generate_hillclimb_mutations(curr_seq, num_mutations: int, alphabet: str, variable_regions: list=None):
    """
    Generate all k=1 mutations and random k=2 mutations

    Args:
        curr_seq: Current sequence
        num_mutations: Number of mutations
        alphabet: Protein alphabet to use
        variable_regions: List of variable regions

    Returns:
        Generated targets
    """
    if variable_regions:
        variable_token_ind = variable_regions
    else:
        variable_token_ind = list(range(len(sequences)))

    targets = []
    # generate all k=1 mutations
    for ind in variable_token_ind:
        pre_string = curr_seq[:ind]
        post_string = curr_seq[ind+1:]
        _alphabet = set(alphabet)-set([curr_seq[ind]])
        targets += [pre_string+letter+post_string for letter in _alphabet]

    # generate random k=2 mutations
    num_k2 = num_mutations-len(targets)
    if num_k2>0:
        for i in range(num_k2):
            inds = random.sample(variable_token_ind,2)
            seq_copy = curr_seq
            for ind in inds:
                seq_copy = seq_copy[:ind]+random.sample(set(alphabet)-set([curr_seq[ind]]),1)[0]+seq_copy[ind+1:]
            targets += [seq_copy]
    return targets