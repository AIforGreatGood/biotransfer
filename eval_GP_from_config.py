# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

"""Hook for evaluating GP with config file"""

import hydra
from random import randint
from src import eval_gp

@hydra.main()
def eval_from_config(cfg):
    return eval_gp(**cfg)

if __name__ == "__main__":
    eval_from_config()
