# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from CovidDataset import CovidDataset
from Sampler import Sampler

train_dataset = CovidDataset(chain='14L',split='train',correction='replicate')
valid_dataset = CovidDataset(chain='14L',split='valid',correction='replicate')

sampler = Sampler([train_dataset, valid_dataset])
for i in range(2000,9000,500):
	sampler.uniform_sample(n=i, write=True, do_valid=True)
