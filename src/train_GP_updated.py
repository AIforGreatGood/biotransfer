# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

"""Training classes and functions"""

from importlib import import_module
import hydra
from torch.utils.data import DataLoader
import torch
import tqdm
import gpytorch
import gc
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
import joblib
#from .LBFGS import FullBatchLBFGS

def find_best_gpu_setting(model_cfg, train_x,
                          train_y,
                          n_devices,
                          output_device,
                          preconditioner_size
):
    N = train_x.size(0)

    # Find the optimum partition/checkpoint size by decreasing in powers of 2
    # Start with no partitioning (size = 0)
    settings = [0] + [int(n) for n in np.ceil(N / 2**np.arange(1, np.floor(np.log2(N))))]

    for checkpoint_size in settings:
        print('Number of devices: {} -- Kernel partition size: {}'.format(n_devices, checkpoint_size))
        try:
            # Try a full forward and backward pass with this setting to check memory usage
            _, _ = train(model_cfg, train_x, train_y,
                         n_devices=n_devices, output_device=output_device,
                         checkpoint_size=checkpoint_size,
                         preconditioner_size=preconditioner_size, n_training_iter=1)

            # when successful, break out of for-loop and jump to finally block
            break
        except RuntimeError as e:
            print('RuntimeError: {}'.format(e))
        except AttributeError as e:
            print('AttributeError: {}'.format(e))
        finally:
            # handle CUDA OOM error
            gc.collect()
            torch.cuda.empty_cache()
    return checkpoint_size

def train(model_cfg,
          train_x,
          train_y,
          n_devices,
          output_device,
          checkpoint_size,
          preconditioner_size,
          n_training_iter,
):

    target_args = model_cfg._target_.split(".")
    module_path = ".".join(target_args[:-1])
    module_name = target_args[-1]
    module = getattr(import_module(module_path), module_name)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if model_cfg.inducing_points:
        inducing_points=min([model_cfg.inducing_points, len(train_X)])
        gp_model = module(train_x, train_y, likelihood, inducing_points=inducing_points)
    else:
        gp_model = module(train_x, train_y, likelihood, n_devices)
    if torch.cuda.is_available():
        gp_model.to(output_device)
        likelihood.to(output_device)
    print(gp_model)

    # -------TRAIN------
    gp_model.train()
    likelihood.train()
    #optimizer = FullBatchLBFGS(gp_model.parameters(), lr=model_cfg.lr)
    optimizer = torch.optim.Adam(gp_model.parameters(), lr=model_cfg.lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
    epoch_iter = tqdm.tqdm(range(model_cfg.num_epochs))

    """
    for i in epoch_iter:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = gp_model(train_X)
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y)
        loss.backward()
        #iterator.set_postfix(loss=loss.item())
        optimizer.step()
        epoch_iter.set_postfix(loss=loss.item())
    """

    with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
         gpytorch.settings.max_preconditioner_size(preconditioner_size):

        def closure():
            optimizer.zero_grad()
            output = gp_model(train_x)
            loss = -mll(output, train_y)
            return loss

        loss = closure()
        loss.backward()

        for i in epoch_iter:
            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
            loss, _, _, _, _, _, _, fail = optimizer.step(options)

            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, n_training_iter, loss.item(),
                gp_model.covar_module.module.base_kernel.lengthscale.item(),
                gp_model.likelihood.noise.item()
            ))

            if fail:
                print('Convergence reached!')
                break

    print(f"Finished training on {train_X.size(0)} data points using {n_devices} GPUs.")
    return gp_model, likelihood

def validation(model, val_dataloader, feat_model, mll, variable_regions=None, feat_concat=False, pca_model=None):
    """Evaluate model on validation set

    Args:
        model: Model to perform validation
        val_dataloader: Validation dataloader
        feat_model: Model to extract features
        mll: Marginal log likelihood loss function
        variable regions: ???
        pca_model: Model to perform pca

    Returns
        Average validation loss
    """
    means = torch.tensor([0.])
    gt = torch.tensor([0.])
    val_loss = []
    with torch.no_grad():
        for batch in tqdm.tqdm(val_dataloader):
            # extract features
            if torch.cuda.is_available():
                data, mask, target = batch["input_ids"].cuda(), batch["input_masks"].cuda(), batch["targets"].cuda()
            else:
                data, mask, target = batch["input_ids"], batch["input_masks"], batch["targets"]
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
    return sum(val_loss)/len(val_loss)

def train_gp(train_set_cfg, train_dataloader_cfg, feat_cfg, model_cfg=None, val_set_cfg=None, val_dataloader_cfg=None, logger_cfgs=None,
         callback_cfgs=None, checkpoint_callback_cfg=None, seed=0, reload_checkpoint_path=None, reload_state_dict_path=None,
         strict_reload=True,
         experiment_name=None, 
         nodes=None):
    """
    Train using given configurations.

    Args:
        train_set_cfg: Training dataset configuration
        train_dataloader_cfg: Training dataloader configuration
        feat_cfg: Feature extractor model configuration
        model_cfg: Model configuration
        val_set_cfg: Validation dataset configuration
        logger_cfgs: List of logger configurations
        callback_cfgs: List of callback configurations
        checkpoint_callback_cfg: Checkpoing callback configuration
        seed: Seed for initialization and training.
        reload_checkpoint_path: Path for reloading model from a
          pytorch-lightning checkpoint
        reload_state_dict_path: Path for reloading model from a
          pytorch state dictionary
        strict_reload: Whether or not the current model must
          exactly match the reloaded weight architecture
        experiment_name: Name of this experiment
        nodes: Number of nodes
    """ 
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

    # prepare train data
    feat_model.eval()
    train_X = torch.tensor([])
    train_y = torch.tensor([])
    if torch.cuda.is_available():
        output_device = torch.device('cuda:0')
        feat_model.to(output_device)
        train_X, train_y = train_X.to(output_device), train_y.to(output_device)
    with torch.no_grad():
        for batch in tqdm.tqdm(train_dataloader):
            if torch.cuda.is_available():
                data, mask, target = batch["input_ids"].to(output_device), batch["input_masks"].to(output_device), batch["targets"].to(output_device)
            else:
                data, mask, target = batch["input_ids"], batch["input_masks"], batch["targets"]
            train_y = torch.cat((train_y, target.squeeze(-1)), 0)
            output = feat_model(data, mask, variable_regions=variable_regions, feat_concat=feat_concat)
            train_X = torch.cat((train_X, output), 0)
    print(train_X.size())
    print(train_y.size())

    if pca_dim is not None: # perform pca
        if torch.cuda.is_available():
            train_X = train_X.cpu()
        pca_model = KernelPCA(pca_dim, kernel='linear', copy_X=False).fit(train_X)
        train_X = torch.from_numpy(pca_model.transform(train_X))
        if torch.cuda.is_available():
            train_X = train_X.to(output_device)
        print(train_X.size())
    else:
        pca_model = None

    # load GP model
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        print('Planning to run on {} GPUs.'.format(n_devices))
        # Set a large enough preconditioner size to reduce the number of CG iterations run
        preconditioner_size = 100
        checkpoint_size = find_best_gpu_setting(model_cfg,
                                                train_X, train_y,
                                                n_devices=n_devices,
                                                output_device=output_device,
                                                preconditioner_size=preconditioner_size)


    """
    # ------Validation------
    if val_set_cfg is not None:
        gp_model.eval()
        likelihood.eval()
        val_loss = validation(gp_model, val_dataloader, feat_model, mll, variable_regions=variable_regions, feat_concat=feat_concat, pca_model=pca_model)
        print('validation loss:', val_loss)
    
    # save model
    torch.save(gp_model.state_dict(), 'GP_model_state_dict.pth')
    if pca_dim is not None: # pca is used and save the pca model
        joblib.dump(pca_model, 'pca_model.sav')
    return val_loss.item()
    """
