# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).


import hydra
from .landscapes.ProxyLandscape import ProxyLandscape
from .utils import parse_env4lightning

def design_pipeline(explorer_cfg, model_cfg=None, dataloader_cfg=None, ensemble_cfg=None,
    maximize=True, seed_topK: int=None, cdr_region: int=None, experiment_name=None, log_file=None, nodes=None):
    """Pipes configs and constructs objects necessary for design process

    Args:
        explorer_cfg: Explorere config
        model_cfg: Model config
        dataloader_cfg: Datalaoder config
        ensemble_cfg: Ensemble config
        maximize: Boolean to maximize or minimize
        seed_topK: Seed for top k
        experiment_name: Experiment name
        log_file: File name to log to
        nodes: Number of nodes to use
    """

    #parse_env4lightning(verbose=True)

    # construct design landscape that takes a sequence and outputs a model score, mean and std of predicted binding
    if ensemble_cfg is None:
        # ensemble model takes a list of pretrained regression model via model finetuning to compute the mean and std of the predicted binding
        assert model_cfg is not None
        assert dataloader_cfg is not None
        # get dataloader
        dataloader = hydra.utils.instantiate(dataloader_cfg)
        seed_sequence, seed_value = dataloader.get_seed_sequence(seed_topK)
        variable_regions = dataloader.get_variable_regions(cdr_region=cdr_region)
        # set up proxy landscape
        print('Loading the model ...')
        model = hydra.utils.instantiate(model_cfg, dataloader=dataloader, variable_regions=variable_regions, seed_value=seed_value)
    else:
        # gaussian process directly outputs the mean and std
        model = hydra.utils.instantiate(ensemble_cfg,seed_topK=seed_topK,cdr_region=cdr_region)
        seed_sequence =  model.seed_sequence
        variable_regions = model.variable_regions
    print('seed_sequence', seed_sequence)
    print('variable_regions',variable_regions)

    # Create landscape using the pretrained regression model with mean and std characterization
    model = ProxyLandscape(model=model, maximize=maximize)

    # set up sampling algorithm
    print('Initializing optimization explorer ...')
    explorer = hydra.utils.instantiate(explorer_cfg, 
                                        model=model, 
                                        starting_sequence=seed_sequence,
                                        variable_regions=variable_regions)

    # run optimization and results are saved in the predefined location specified by the config file
    print('Performing optimization ...')
    results = explorer.run(landscape=None)