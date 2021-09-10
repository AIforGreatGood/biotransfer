# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
from pathlib import Path
import math

class Sampler():
    def __init__(self,datasets,seed=0):
        if not isinstance(datasets, list):
             datasets = list(datasets)
        self.datasets = datasets
        self.data_path = datasets[0].data_path
        self.sampled_data = []
        self.seed = seed
        np.random.seed(seed)

    def do_uniform_sampling(self,dataset, n):
        inds = list(range(len(dataset)))
        random_inds = np.random.choice(inds, n)
        sel_data = map(dataset.data.__getitem__, random_inds)
        self.sampled_data.append(sel_data)
        sel_data_df = pd.DataFrame.from_records(list(sel_data))
        return sel_data_df
    

    def uniform_sample(self,n=200, write=True, do_valid=True):
         for dataset in self.datasets:
              _n = n
              if do_valid and dataset.split == "valid":
                  _n = math.ceil(0.15*n)
              sel_data_df = self.do_uniform_sampling(dataset,_n)
              if write:
                  dir_path = self.data_path[:-1]+"_random_"+str(n)+ '/{}_corrected/{}/'.format(dataset.correction, dataset.chain)
                  Path(dir_path).mkdir(parents=True, exist_ok=True)
                  sel_data_df.to_csv(dir_path+'{}_{}.csv'.format(dataset.chain, dataset.split))
         return self.sampled_data

