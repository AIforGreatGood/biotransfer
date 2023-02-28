# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

"""Training classes and functions"""

from importlib import import_module
import os
from random import random
import time

import hydra
import pytorch_lightning
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader
import torch

#from .utils import parse_env4lightning

def train(train_set_cfg, train_dataloader_cfg, trainer_cfg, model_cfg=None, val_set_cfg=None, val_dataloader_cfg=None, logger_cfgs=None,
         callback_cfgs=None, checkpoint_callback_cfg=None, seed=0, reload_checkpoint_path=None, reload_state_dict_path=None,
         strict_reload=True,
         experiment_name=None, 
         nodes=None):
    """Train using given configurations.

    Args:
        train_set_cfg: Training dataset omegaconf configuration
        train_dataloader_cfg: Training dataloader omegaconf configuration
        trainer_cfg: Pytorch-lightning trainer omegaconf configuration
        model_cfg: Model omegaconf configuration
        val_set_cfg: Validation dataset omegaconf configuration
        val_dataloader_cfg: Validation dataloader omegaconf configuration
        logger_cfgs: Logger omegaconf configurations
        checkpoint_callback_cfg: Checkpoint callback omegaconf configuration
        seed: Random seed for program.
        reload_checkpoint_path: Path for reloading a model from a
            pytorch-lightning checkpoint
        reload_state_dict_path: Path for reloading a model from a 
            pytorch state_dict
        strict_reload: Whether or not the reloaded architecture must exactly
            match current architecture.
        experiment_name: The name of the experiment.
        nodes: The number of slurm nodes needed.

    Returns
        Validation loss if it exists, None otherwise.
    """

    #parse_env4lightning(verbose=True)
    seed_everything(seed=seed)
        
    # Load training data handlers
    train_set = hydra.utils.instantiate(train_set_cfg)
    print("TRAIN SET SIZE: {}".format(len(train_set)))
    if hasattr(train_set, "collate_fn"):
        train_dataloader = DataLoader(dataset=train_set, collate_fn=train_set.collate_fn, **train_dataloader_cfg)
    else:
        train_dataloader = DataLoader(dataset=train_set, **train_dataloader_cfg)
    
    # Load validation data handlers
    val_dataloader=None
    if val_set_cfg is not None:
        val_set = hydra.utils.instantiate(val_set_cfg)
        if hasattr(val_set, "collate_fn"):
            val_dataloader = DataLoader(dataset=val_set, collate_fn=val_set.collate_fn, **val_dataloader_cfg)
        else:
            val_dataloader = DataLoader(dataset=val_set, **val_dataloader_cfg)

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
        for key in callback_cfgs:
            callbacks.append(hydra.utils.instantiate(callback_cfgs[key]))
            
    checkpoint_callback = True
    if checkpoint_callback_cfg is not None:
        checkpoint_callback = ModelCheckpoint(**checkpoint_callback_cfg)
            
    trainer = Trainer(**trainer_cfg, logger=loggers, callbacks=callbacks,
                         checkpoint_callback=checkpoint_callback, deterministic=True)
    
    # Fit
    start_time = time.time()
    #breakpoint()
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

    print("Elapsed training time: {}".format(time.time() - start_time))

    if val_set_cfg is None:
        return
    else:
        print("Final validation loss: {}".format(model.val_loss))
        return model.val_loss
