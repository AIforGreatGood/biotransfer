# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import gpytorch
import numpy as np

class GPRegressionModel(gpytorch.models.ExactGP):
    """Exact Gaussian Process for regression"""
    def __init__(self, train_x, train_y, likelihood, lengthscale=30):
        """Inits model. See gpytorch for details
        Args:
            train_x: Training dataset input array
            train_y: Training dataset label array
            likelihood: Likehood model from gpytorch
        """
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.constant.data[:] = train_y.mean().item()
            
        self.covar_module = self.base_covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))
        self.covar_module.base_kernel.lengthscale = lengthscale
        
        v = train_y.var().item()
        likelihood.noise = v/2
        self.covar_module.outputscale = v/2

    def forward(self, x):
        """Forward hook
        Args:
            x: Inference samples
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPSparse(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood, lengthscale=30, inducing_points=500):
        super(GPSparse, self).__init__(x, y, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.mean.constant.data[:] = y.mean().item()
            
        self.base_covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=x.shape[1]))
        self.base_covar.base_kernel.lengthscale = lengthscale
        
        v = y.var().item()
        likelihood.noise = v/2
        self.base_covar.outputscale = v/2
        
        random = np.random.RandomState(1)
        select = random.permutation(len(x))[:inducing_points]
        points = x[select]
        self.covar = gpytorch.kernels.InducingPointKernel(self.base_covar, inducing_points=points, likelihood=likelihood)

    def forward(self, x):
        mu = self.mean(x)
        cov = self.covar(x)
        return gpytorch.distributions.MultivariateNormal(mu, cov)
