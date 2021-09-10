# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""Hook for training non-GP model with config file"""

import hydra
from random import randint
from src import train

@hydra.main()
def train_from_config(cfg):
    return train(**cfg)

if __name__ == "__main__":
    train_from_config()
