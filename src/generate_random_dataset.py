# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""Generates a random dataset of peptide strings"""

from src.datasets.RandomPeptideGenerator import RandomPeptideGenerator
import json


if __name__ == "__main__":
	with open('/home/gridsan/ES26698/lightning_distributed/train_hist_example.json','r') as json_file:
	    heavychain_train_length_histogram = json.load(json_file)
	generator = RandomPeptideGenerator(seed=24) #train used seed = 30, valid used seed = 24

	data_directory = '/home/gridsan/ES26698/biotransfer/data/random_peptide'

	#num_of_peptides = 32593668 #number of samples in pfam_train
	num_of_peptides = 1715454 #number of samples in pfam_valid

	print('Generating '+str(num_of_peptides)+' random peptides')

	generator.create_lmdb(data_directory,num_of_peptides,length_histogram=heavychain_train_length_histogram, split="valid")

	print('Generated Random Peptide Dataset!')
