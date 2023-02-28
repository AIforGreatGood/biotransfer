# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

"""Run design with config file"""

import hydra
from src.design_pipeline import design_pipeline
import hydra.experimental

#config_dir = '/home/li25662/AIforGreatGood/biotransfer/configs/design_configs'
#hydra.initialize_config_dir(config_dir)
#cfg = hydra.compose("design_pipeline_gp_14H_hc.yaml")

@hydra.main()
def design_from_config(cfg):
    return design_pipeline(**cfg)

if __name__ == "__main__":
    design_from_config()
