# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

import importlib
import os

import hydra
import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
from scipy.stats import pearsonr, spearmanr
from .modeling_bert import SimpleMLP

# +
class GiffordCNN(LightningModule):
    """
    GiffordCNN for regression.
    """

    def __init__(self, enc_channels, enc_kernel_sizes, enc_strides, enc_paddings,
        enc_hidden_sizes,
        mp_kernel_size,
        mp_stride,
        dropout=0,
        optimizer_cfg=None, verbose=False,
        inducing_points=None):

        super().__init__()

        self.save_hyperparameters()

        self.enc_channels = enc_channels
        self.enc_kernel_sizes = enc_kernel_sizes
        self.enc_strides = enc_strides
        self.enc_paddings = enc_paddings
        self.enc_hidden_sizes = enc_hidden_sizes
        self.mp_kernel_size = mp_kernel_size
        self.mp_stride = mp_stride
        self.dropout = dropout
        self.optimizer_cfg = optimizer_cfg
        self.verbose = verbose

        # Encoder convolution
        self.enc_conv_layer_list = []
        for i in range(len(self.enc_channels)-1):

            ins = self.enc_channels[i]
            outs = self.enc_channels[i+1]

            # Convolution
            layer = nn.Conv1d(in_channels=ins, out_channels=outs,
                kernel_size=self.enc_kernel_sizes[i], stride=self.enc_strides[i], padding=self.enc_paddings[i])
            self.enc_conv_layer_list.append(layer)

            # Max pool
            self.enc_conv_layer_list.append(nn.MaxPool1d(kernel_size=self.mp_kernel_size, stride=self.mp_stride))

            # Activation function
            self.enc_conv_layer_list.append(nn.ReLU())
        self.enc_conv_layer_list = nn.ModuleList(self.enc_conv_layer_list)

        # Encoder dnn
        self.output_layer = SimpleMLP(self.enc_hidden_sizes[0],self.enc_hidden_sizes[1],1,dropout)
        
        self.enc_dnn_list = []
        """
        if self.enc_hidden_sizes:
            for i in range(len(self.enc_hidden_sizes)-1):

                # Linear
                layer = nn.Linear(self.enc_hidden_sizes[i], self.enc_hidden_sizes[i+1])
                self.enc_dnn_list.append(layer)

                # Activation function
                if i != len(self.enc_hidden_sizes)-2:
                    self.enc_dnn_list.append(nn.ReLU())

                # Dropout
                if self.dropout is not None:
                    self.enc_dnn_list.append(nn.Dropout(p=self.dropout))
            self.enc_dnn = nn.Sequential(*self.enc_dnn_list)

        # Output
        self.output_layer = nn.Linear(self.enc_hidden_sizes[0], self.enc_hidden_sizes[-1], 1)
        """
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        latent, conv_shape = self.encode(x)
        #latent_flat = torch.flatten(latent, start_dim=1)
        output = self.output_layer(latent)
        if self.verbose: print("Input: {}".format(x.shape))
        if self.verbose: print("Output: {}".format(output.shape))
        return output

    def encode(self, x):

        # Loop through CNN encoders
        for i in range(len(self.enc_conv_layer_list)):
            if i == 0:
                enc_conv_out = x

            if self.verbose: print("Enc conv out {}: {}".format(i, enc_conv_out.shape))
            enc_conv_out = self.enc_conv_layer_list[i](enc_conv_out)

        conv_shape = enc_conv_out.shape
        if self.verbose: print("Enc flat: {}".format(enc_conv_out.shape))

        # Loop through DNN encoders
        # If dense list is empty, don't use
        if self.enc_dnn_list:
            #print(enc_conv_out.shape)
            enc_conv_out = torch.flatten(enc_conv_out, start_dim=1)
            enc_dnn_out = self.enc_dnn(enc_conv_out)
        else:
            enc_dnn_out = torch.flatten(enc_conv_out, start_dim=1)

        return enc_dnn_out, conv_shape
    
    def configure_optimizers(self):
        if self.optimizer_cfg is None:
            raise Exception("optimizer_cfg not defined in this model. Check the config file.")
        return hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())
    
    def sync_across_gpus(self, t):   # t is a tensor
        """Aggregate results from across gpus. See Lin for details

        Args:
            t: A tensor.

        Returns:
            All collected instances of tensor across gpus?
        """
        # a work-around function to sync outputs across multiple gpus to compute a metric
        gather_t_tensor = [torch.ones_like(t) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather(gather_t_tensor, t)
        return torch.cat(gather_t_tensor)
    
    def training_step(self, batch, batch_idx):
        x = batch["input_ids"]
        y = batch["targets"]

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        if self.trainer.global_step % 25 == 0:
            self.logger.experiment.add_scalar("Train/loss", loss.detach(), self.trainer.global_step)

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        metrics = {"loss": avg_loss}
    
    def validation_step(self, batch, batch_idx):
        x = batch["input_ids"]
        y = batch["targets"]

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        #return {'val_loss': loss.detach()}
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        metrics = {"val_loss": avg_loss}
        self.val_loss = avg_loss.item()

        print("Val loss: {}".format(avg_loss.detach()))

        self.logger.experiment.add_scalar("Valid/loss", avg_loss.detach(), self.trainer.global_step)
        return metrics
    
    def test_step(self, batch, batch_idx):
        x = batch["input_ids"]
        y = batch["targets"]

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        return {'test_loss': loss.detach(), "preds": y_hat.squeeze(1), "targets": y.squeeze(1)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        metrics = {"test_loss": avg_loss}
        self.test_loss = avg_loss.item()

        print("Test loss: {}".format(avg_loss.detach()))
        print('Output keys', outputs[0].keys())
        self.logger.experiment.add_scalar("Test/loss", avg_loss.detach(), self.trainer.global_step)
        preds = torch.cat([x['preds'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])
        
        if self.trainer.use_ddp:
            preds_all = self.sync_across_gpus(preds)
            targets_all = self.sync_across_gpus(targets)
            predictions_all = preds_all.cpu().detach().numpy()
            targets_all = targets_all.cpu().detach().numpy()
            pearsonr_correlation = pearsonr(targets_all, predictions_all)[0]
            spearmanr_metric = spearmanr(targets_all, predictions_all)[0]
            print('pearson correlation (all)', pearsonr_correlation)
            print('spearman rho (all)', spearmanr(targets_all, predictions_all)[0])
            print('loss (all)', ((targets_all - predictions_all)**2).mean())
            
            
            
