# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

"""Eval classes and functions"""

import hydra
import numpy as np
import torch
import warnings
from torch.utils.data import DataLoader
import torch.nn.functional as F
from statistics import NormalDist
import gpytorch
from src.models.ExactGP import GPRegressionModel
from src.models.feat_extraction import load_featmodel_from_checkpoint
from torch.serialization import SourceChangeWarning

import onnx, onnxruntime
import tqdm
import joblib
hydra.core.global_hydra.GlobalHydra().clear()

class EnsembleOnnxModelPrediction():
    """Ensemble model"""

    def __init__(self, model_cfg, dataloader_cfg, batch_size: 16, cdr_region:int=None, seed_topK:int=None, additional_models=None):
        """
        Args:
            model_cfg: Model config
            dataloader_cfg: Dataloader config
            batch_size: Batch size for model
            seed_topK: Seed for top k
            additional_models: ???
        """
        self.batch_size = batch_size
        # load dataloader
        self.dataset = hydra.utils.instantiate(dataloader_cfg)
        self.seed_sequence, self.seed_value = self.dataset.get_seed_sequence(seed_topK)
        self.variable_regions = self.dataset.get_variable_regions(cdr_region=cdr_region)
        
        # load pretrained model
        self.ort_sessions = []
        for model_path in model_cfg.onnx_model_paths:
            print('Loading model', model_path)
            self.ort_sessions.append(onnxruntime.InferenceSession(model_path))

        # load additional models
        self.additional_ort_sessions = []
        if additional_models is not None:
            model_cfg = additional_models.model_cfg
            dataloader_cfg = additional_models.dataloader_cfg
            self.additional_dataset = hydra.utils.instantiate(dataloader_cfg)
            
            for model_path in model_cfg.onnx_model_paths:
                print('Loading additional model', model_path)
                self.additional_ort_sessions.append(onnxruntime.InferenceSession(model_path))

    def predict(self, sequences: list):
        """
        Compute the prediction score

        Args:
            sequences: List of sequences

        Returns
            The log cumulative distribution function and array of means and
            standard deviations.
        """
        predictions_all_models = []

        if self.ort_sessions:
            self.dataset.add_data(sequences)
            # dataloader
            if hasattr(self.dataset, "collate_fn_onnx"):
                dataloader = DataLoader(dataset=self.dataset, collate_fn=self.dataset.collate_fn_onnx, batch_size=self.batch_size)
            else:
                dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size)
            # predict
            for ort_session in self.ort_sessions:
                predictions = []
                for data in dataloader:
                    ort_outs = ort_session.run(None, data)
                    predictions.extend(ort_outs[0].T[0]) 
                predictions_all_models.append(predictions)

        if self.additional_ort_sessions:
            self.additional_dataset.add_data(sequences)
            # dataloader
            if hasattr(self.additional_dataset, "collate_fn_onnx"):
                additional_dataloader = DataLoader(dataset=self.additional_dataset, collate_fn=self.additional_dataset.collate_fn_onnx, batch_size=self.batch_size)
            else:
                additional_dataloader = DataLoader(dataset=self.additional_dataset, batch_size=self.batch_size)
            # predict
            for ort_session in self.additional_ort_sessions:
                predictions = []
                for data in additional_dataloader:
                    ort_outs = ort_session.run(None, data)
                    predictions.extend(ort_outs[0].T[0]) # logit
                predictions_all_models.append(predictions)

        predictions_all_models = np.array(predictions_all_models)
        mean = predictions_all_models.mean(axis=0)
        std = predictions_all_models.std(axis=0)
        cdf = []
        for mu,sigma in list(zip(mean,std)):
            if sigma == 0:
                sigma = 2.220446049250313e-16
            cdf.append(NormalDist(mu=mu, sigma=sigma).cdf(self.seed_value))
        log_cdf = np.log(np.array(cdf))
        #print('mean', mean)
        #print('std', std)
        return log_cdf, np.array(mean), np.array(std)

class GPModelPrediction():
    """Gaussian process model"""

    def __init__(self, feat_cfg, pca_model_path, gp_state_dict_path, dataloader, batch_size: int=16,variable_regions=None,seed_value=None):
        """
        Args:
            feat_cfg: Feature config
            pca_model_path: Path to the pca model saved state
            gp_state_dict_path: Path to the gaussian process saved state
              dictionary
            dataloader: Dataloader
            batch_size: Batch size
            variable_regions: Variable regions in sequences
            seed_value: Seed value
        """
        print('feat_cfgfeat_cfgfeat_cfgfeat_cfgfeat_cfg',feat_cfg)
        self.seed_value = seed_value
        self.batch_size = batch_size
        self.variable_regions = np.array(variable_regions)+1 # +1 to offset the starting token in the language model
        # get dataloader
        self.dataset = dataloader
        print('load feature model')
        # load feature model
        self.feat_model = load_featmodel_from_checkpoint(model_config_file=feat_cfg.model_config_file,feat_path=feat_cfg.feat_path,strict_reload=feat_cfg.strict_reload)
        """
        target_args = feat_cfg._target_.split(".")
        module_path = ".".join(target_args[:-1])
        print('module path',module_path)
        module_name = target_args[-1]
        print('module name',module_name)
        module = getattr(import_module(module_path), module_name)
        print('module', module)
        self.feat_model = module.load_from_checkpoint(feat_cfg.feat_path, model_config_file=feat_cfg.model_config_file, strict=feat_cfg.strict_reload)
        self.feat_model.eval()
        """
        if torch.cuda.is_available():
            self.feat_model.cuda()
        print('load pca model')
        self.pca_model = joblib.load(pca_model_path)
        print('load gp model (gp model currently does not support onnx)')
        X, y = self.prepare_labeled_data()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if torch.cuda.is_available():
            X.cuda()
            y.cuda()
        self.gp_model = GPRegressionModel(X, y, likelihood)
        self.gp_model.load_state_dict(torch.load(gp_state_dict_path))
        self.gp_model.eval()
        if torch.cuda.is_available():
            self.gp_model.cuda()
            likelihood.cuda()

    def prepare_labeled_data_(self):
        if hasattr(self.dataset, "collate_fn_onnx"):
            dataloader = DataLoader(dataset=self.dataset, collate_fn=self.dataset.collate_fn_onnx, batch_size=self.batch_size)
        else:
            dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size)
        X = []
        y = []
        for data in dataloader:
            y.extend(data['targets'].tolist())
            ort_outs = self.feat_ort_session.run(None, {'input_ids': data['input_ids'], 'input_mask': data['input_mask'], 'variable_regions': self.variable_regions})
            X.extend(ort_outs[0])
        X = np.array(X)
        y = np.array(y).squeeze() 
        # dimensionality reduction via pretrained pca
        X = torch.from_numpy(self.pca_model.transform(X))
        y = torch.tensor(y)
        return X, y

    def prepare_labeled_data(self):
        if hasattr(self.dataset, "collate_fn"):
            dataloader = DataLoader(dataset=self.dataset, collate_fn=self.dataset.collate_fn, batch_size=self.batch_size)
        else:
            dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size)
        if torch.cuda.is_available():
            train_X = torch.tensor([]).cuda()
            train_y = torch.tensor([]).cuda()
        with torch.no_grad():
            for batch in dataloader:
                if torch.cuda.is_available():
                    data, mask, target = batch["input_ids"].cuda(), batch["input_mask"].cuda(), batch["targets"].cuda()
                train_y = torch.cat((train_y, target.squeeze(-1)), 0)
                output = self.feat_model(data, mask, variable_regions=self.variable_regions)
                train_X = torch.cat((train_X, output), 0)
        # dimensionality reduction via pretrained pca
        if torch.cuda.is_available():
            train_X = train_X.cpu()
        train_X = torch.from_numpy(self.pca_model.transform(train_X))
        if torch.cuda.is_available():
            train_X = train_X.cuda()
        return train_X, train_y

    def predict(self, sequences: list):
        """
        Compute the prediction score

        Args:
            sequences: List of sequences

        Returns
            Predictions
        """
        self.dataset.add_data(sequences)

        if hasattr(self.dataset, "collate_fn"):
            dataloader = DataLoader(dataset=self.dataset, collate_fn=self.dataset.collate_fn, batch_size=self.batch_size)
        # extract features
        means = torch.tensor([])
        stds = torch.tensor([])
        with torch.no_grad():
            for batch in dataloader:
                # extract features
                if torch.cuda.is_available():
                    data, mask = batch["input_ids"].cuda(), batch["input_mask"].cuda()
                val_X = self.feat_model(data, mask, variable_regions=self.variable_regions)
                if torch.cuda.is_available():
                    val_X = val_X.cpu()
                val_X = torch.from_numpy(self.pca_model.transform(val_X))
                if torch.cuda.is_available():
                    val_X = val_X.cuda()
                output = self.gp_model(val_X)  
                means = torch.cat([means, output.mean.cpu()])
                stds = torch.cat([stds, output.stddev.cpu()])

        means = means.numpy()
        stds = stds.numpy()

        cdf = []
        for mu,sigma in list(zip(means,stds)):
            if sigma == 0:
                sigma = 2.220446049250313e-16
            cdf.append(NormalDist(mu=mu, sigma=sigma).cdf(self.seed_value))
        log_cdf = np.log(np.array(cdf))
        return log_cdf, np.array(means), np.array(stds)


