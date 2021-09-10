# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""Training classes and functions"""

from importlib import import_module
import hydra
from torch.utils.data import DataLoader
import torch
import tqdm
import gpytorch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
import joblib

def evaluation(model, val_dataloader, feat_model, mll, variable_regions=None, pca_model=None, feat_concat=False):
    """Perform model evaluation

    Args:
        model: Model to evaluate
        val_dataloader: Validation dataloader
        feat_model: Feature extractor model
        mll: Marginal loss likelihood loss function
        variable_regions: ???
        pca_model: Model to perform pca

    Returns
        Mean validation loss, prediction means, ground truth (?)
    """
    means = torch.tensor([0.])
    gt = torch.tensor([0.])
    val_loss = []
    with torch.no_grad():
        for batch in tqdm.tqdm(val_dataloader):
            # extract features
            if torch.cuda.is_available():
                data, mask, target = batch["input_ids"].cuda(), batch["input_masks"].cuda(), batch["targets"].cuda()
            val_X = feat_model(data, mask, variable_regions=variable_regions, feat_concat=feat_concat)
            if pca_model is not None:
                if torch.cuda.is_available():
                    val_X = val_X.cpu()
                val_X = torch.from_numpy(pca_model.transform(val_X))
                if torch.cuda.is_available():
                    val_X = val_X.cuda()
            output = model(val_X)  
            means = torch.cat([means, output.mean.cpu()])
            gt = torch.cat([gt, target.squeeze(-1).cpu()])
            val_loss.append(-mll(output, target.squeeze(-1)))
    gt = gt[1:]
    means = means[1:]
    print('Test MAE: {}'.format(torch.mean(torch.abs(means - gt))))
    gt = gt.detach().cpu().numpy()
    means = means.detach().cpu().numpy()
    print('Test Pearson: {}'.format(pearsonr(gt, means)[0]))
    return sum(val_loss)/len(val_loss), means, gt

def eval_gp(model_cfg, feat_cfg, train_set_cfg, train_dataloader_cfg, eval_set_cfg, eval_dataloader_cfg, 
val_set_cfg=None, val_dataloader_cfg=None, seed=0, reload_state_dict_path=None, pca_model_path=None,
strict_reload=True,eval_function_cfg=None,experiment_name=None, nodes=None):
    """
    Eval using given configurations.

    Args:
        model_cfg: Model omegaconf configuration
        feat_cfg:
        train_set_cfg: Training dataset omegaconf configuration
        train_dataloader_cfg: Training dataloader omegaconf configuration
        eval_set_cfg: Evaluation dataset omegaconf configuration
        eval_dataloader_cfg: Evaluation dataloader omegaconf configuration
        val_set_cfg: validation dataset omegaconf configuration
        val_dataloader_cfg: validation dataloader omegaconf configuration
        seed: Seed for initialization and training
        reload_state_dict_path: Path for reloading the model state dictionary
        pca_model_path: Path for reloading the pca model
        strict_reload: Whether or not the state dictionary must share all
          parameters with the current model
        eval_function_cfg: Evaluation function to use
        experiment_name: Name of experiment
        nodes: Number of nodes
    """ 
    # Load data handlers
    train_set = hydra.utils.instantiate(train_set_cfg)
    if hasattr(train_set, "collate_fn"):
        train_dataloader = DataLoader(dataset=train_set, collate_fn=train_set.collate_fn, **train_dataloader_cfg)
    else:
        train_dataloader = DataLoader(dataset=train_set, **train_dataloader_cfg)
    
    if val_set_cfg is not None:
        val_set = hydra.utils.instantiate(val_set_cfg)
        if hasattr(val_set, "collate_fn"):
            val_dataloader = DataLoader(dataset=val_set, collate_fn=val_set.collate_fn, **val_dataloader_cfg)
        else:
            val_dataloader = DataLoader(dataset=val_set, **val_dataloader_cfg)

    eval_set = hydra.utils.instantiate(eval_set_cfg)
    if hasattr(eval_set, "collate_fn"):
        eval_dataloader = DataLoader(dataset=eval_set, collate_fn=eval_set.collate_fn, **eval_dataloader_cfg)
    else:
        eval_dataloader = DataLoader(dataset=eval_set, **eval_dataloader_cfg)

    # Load feature extractor (pretrained language model)
    target_args = feat_cfg._target_.split(".")
    module_path = ".".join(target_args[:-1])
    module_name = target_args[-1]
    module = getattr(import_module(module_path), module_name)
    feat_model = module.load_from_checkpoint(feat_cfg.feat_path, model_config_file=feat_cfg.model_config_file, strict=feat_cfg.strict_reload)
    # pca dimensionality reduction
    pca_dim = feat_cfg.pca_dim
    assert isinstance(pca_dim, int) or (pca_dim is None), 'pca_dim is either None or an positive integer'
    # extract variable_regions indices if variable_regions=True
    if not feat_cfg.variable_regions:
        variable_regions = None
    else:
        variable_regions = train_set.get_variable_regions()
        print('Variable regions:', variable_regions)
    # feature concatenation
    feat_concat = feat_cfg.feat_concat
    if feat_concat: 
        assert pca_dim is not None, 'Please specify an integer value for pca_dim, or set feat_concat to False.'
    if variable_regions is None:
        assert not feat_concat, 'If no variable regions is specified, then features cannot be concatenated. Either set variable_regions=True or set feat_concat=False'

    # prepare labeled data
    feat_model.eval()
    if torch.cuda.is_available():
        feat_model.cuda()
        train_X = torch.tensor([]).cuda()
        train_y = torch.tensor([]).cuda()
    with torch.no_grad():
        for batch in tqdm.tqdm(train_dataloader):
            if torch.cuda.is_available():
                data, mask, target = batch["input_ids"].cuda(), batch["input_masks"].cuda(), batch["targets"].cuda()
            train_y = torch.cat((train_y, target.squeeze(-1)), 0)
            output = feat_model(data, mask, variable_regions=variable_regions, feat_concat=feat_concat)
            train_X = torch.cat((train_X, output), 0)
            train_X_size = len(train_X)
        if val_set_cfg is not None:
            for batch in tqdm.tqdm(val_dataloader):
                if torch.cuda.is_available():
                    data, mask, target = batch["input_ids"].cuda(), batch["input_masks"].cuda(), batch["targets"].cuda()
                train_y = torch.cat((train_y, target.squeeze(-1)), 0)
                output = feat_model(data, mask, variable_regions=variable_regions, feat_concat=feat_concat)
                train_X = torch.cat((train_X, output), 0)
    print(train_X.size())
    print(train_y.size())

    if pca_dim is not None: # perform pca
        if torch.cuda.is_available():
            train_X = train_X.cpu()
        print('load pca model')
        pca_model = joblib.load(pca_model_path)
        #pca_model = KernelPCA(pca_dim, kernel='linear', copy_X=False).fit(train_X)
        train_X = torch.from_numpy(pca_model.transform(train_X))
        if torch.cuda.is_available():
            train_X = train_X.cuda()
        print(train_X.size())
    else:
        pca_model = None

    # load GP model
    target_args = model_cfg._target_.split(".")
    module_path = ".".join(target_args[:-1])
    module_name = target_args[-1]
    module = getattr(import_module(module_path), module_name)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if model_cfg.inducing_points:
        inducing_points=min([model_cfg.inducing_points, train_X_size])
        gp_model = module(train_X, train_y, likelihood, inducing_points=inducing_points)
    else:
        gp_model = module(train_X, train_y, likelihood)
    if reload_state_dict_path is not None:
        gp_model.load_state_dict(torch.load(reload_state_dict_path))
    if torch.cuda.is_available():
        gp_model.cuda()
        likelihood.cuda()
    print(gp_model)

    # ------evaluation------
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
    gp_model.eval()
    likelihood.eval()
    eval_loss, preds, gt = evaluation(gp_model, eval_dataloader, feat_model, mll, variable_regions=variable_regions, pca_model=pca_model, feat_concat=feat_concat)
    print('validation loss:', eval_loss)

    # eval (if evaluation function is provided)
    if eval_function_cfg is not None:
        eval_function = hydra.utils.instantiate(eval_function_cfg)
        eval_function.evaluate(gt, preds)
    