# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

"""Eval classes and functions"""

from importlib import import_module
import os
import time

import hydra
import pytorch_lightning
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader

#from .utils import parse_env4lightning

def eval(model_cfg, eval_set_cfg, eval_dataloader_cfg, trainer_cfg, logger_cfgs=None,
         callback_cfgs=None, seed=0, reload_checkpoint_path=None, reload_state_dict_path=None,
         strict_reload=True,
         experiment_name=None, nodes=None,
         eval_function_cfg=None):
    """Eval using given configurations.

    Args:
        model_cfg: Model omegaconf configuration
        eval_set_cfg: Evaluation dataset omegaconf configuration
        trainer_cfg: Pytorch-lightning trainer omegaconf configuration
        logger_cfgs: Logger omegaconf configurations
        callback_cfgs: Pytorch-lightning callback omegaconf configurations
        seed: Random seed for program.
        reload_checkpoint_path: Path for reloading a model from a
            pytorch-lightning checkpoint
        reload_state_dict_path: Path for reloading a model from a 
            pytorch state_dict
        strict_reload: Whether or not the reloaded architecture must exactly
            match current architecture.
        experiment_name: The name of the experiment.
        nodes: The number of slurm nodes needed.
        eval_function_cfg: Evaluation function omegaconf configuration
    """

    #parse_env4lightning(verbose=True)
    seed_everything(seed=seed)
        
    # Load training data handlers
    eval_set = hydra.utils.instantiate(eval_set_cfg)
    if hasattr(eval_set, "collate_fn"):
        eval_dataloader = DataLoader(dataset=eval_set, collate_fn=eval_set.collate_fn, **eval_dataloader_cfg)
    else:
        eval_dataloader = DataLoader(dataset=eval_set, **eval_dataloader_cfg)
        
    # Load model (loads optimizer if included in configuration)
    if reload_checkpoint_path is not None:
        target_args = model_cfg._target_.split(".")
        module_path = ".".join(target_args[:-1])
        module_name = target_args[-1]
        module = getattr(import_module(module_path), module_name)

        processed_model_cfg = {}
        if model_cfg is not None:
            for key in model_cfg.keys():
                if key != "_target_":
                    processed_model_cfg[key] = model_cfg[key]
        model = module.load_from_checkpoint(reload_checkpoint_path, **processed_model_cfg, strict=strict_reload)

    else:
        model = hydra.utils.instantiate(model_cfg)
        
    if reload_state_dict_path is not None:
        model.load_state_dict(torch.load(reload_state_dict_path))
    print(model)
    
    # Load loggers
    loggers = None
    if logger_cfgs is not None:
        if len(logger_cfgs) == 1:
            loggers = hydra.utils.instantiate(logger_cfgs[0])
        else:
            loggers = []
            for logger_cfg in logger_cfgs:
                loggers.append(hydra.utils.instantiate(logger_cfg))
            
    # Load callbacks
    callbacks = None
    if callback_cfgs is not None:
        callbacks = []
        for callback_cfg in callback_cfgs:
            callbacks.append(hydra.utils.instantiate(callback_cfg))
            
    trainer = Trainer(**trainer_cfg, logger=loggers, callbacks=callbacks)

    # Fit
    start_time = time.time()
    trainer.test(model, test_dataloaders=eval_dataloader)
    print("Elapsed eval time: {}".format(time.time() - start_time))
    print("Final loss: {}".format(model.test_loss))

    # eval (if evaluation function is provided)
    if eval_function_cfg is not None:
        eval_function = hydra.utils.instantiate(eval_function_cfg)
        eval_function.evaluate(model.targets_all, model.predictions_all)




