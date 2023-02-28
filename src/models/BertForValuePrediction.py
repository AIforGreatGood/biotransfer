# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

from tape import utils
from pytorch_lightning import LightningModule
import torch
import numpy as np
from tape.models.modeling_utils import ProteinConfig
from .modeling_bert import ProteinBertForValuePrediction
from scipy.stats import pearsonr, spearmanr


class BertForValuePrediction(LightningModule):
    """Bert model with a value regression head"""

    def __init__(self, model_config_file, lr, downstream_hid_dim, downstream_dropout, loss_function='mse', from_pretrained=None):
        """Inits model

        Args:
            model_config_file: Path to config file
            downstream_hid_dim: Downstream hidden layer dimensionality
            downstream_dropout: Value of downstream dropout
            loss_function: loss function to use
            from_pretrained: Path to pretrained model if necessary
            warmup_steps: linear warmup scheduling for learning rate
        """
        super().__init__()
        self.save_hyperparameters()
        config = ProteinConfig().from_pretrained(model_config_file)
        self.model = ProteinBertForValuePrediction(config, downstream_hid_dim, downstream_dropout, loss_function)
        if from_pretrained is not None:
            self.model = self.model.from_pretrained(from_pretrained)
        self.lr = lr
        self.logits = []
        self.targets = []

    def forward(self, x):
        """Forward hook for model

        Args:
            x: Input batch

        Returns:
            The model outputs
        """
        x = batch["input_ids"]
        mask = batch["input_masks"]
        y = batch["targets"]

        outputs = self.model(x,mask,y)
        return outputs

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
        """Training step for model

        Args:
            batch: A batch of samples to train on
            batch_idx: The index of the current batch
        """
        x = batch["input_ids"]
        mask = batch["input_masks"]
        y = batch["targets"]

        outputs = self.model(x,mask,y)
        if isinstance(outputs[0],tuple):
            loss, metrics = outputs[0]
        else:
            loss = outputs[0]
            metrics = {}

        self.logger.experiment.add_scalar("Train/step/loss", loss, self.trainer.global_step)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        """Called at the end of training epoch.
           See pytorch-lightning for functionality
        """
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        #avg_pearson = torch.stack([x['pearson'] for x in outputs]).mean()
        metrics = {"loss": avg_loss}

        self.logger.experiment.add_scalar("Train/epoch/loss", avg_loss, self.current_epoch)
        #self.logger.experiment.add_scalar("Train/epoch/pearson", avg_pearson, self.current_epoch)

        return 

    def validation_step(self, batch, batch_idx):
        """Validation step for model

        Args:
            batch: A batch of samples to train on
            batch_idx: The index of the current batch
        """
        x = batch["input_ids"]
        mask = batch["input_masks"]
        y = batch["targets"]
        outputs = self.model(x,mask,y)
        if isinstance(outputs[0],tuple):
            loss, metrics = outputs[0]
        else:
            loss = outputs[0]
            metrics = {}
        return {'val_loss': loss, 'logits': outputs[1].squeeze(1), 'targets': y.squeeze(1)}

    def validation_epoch_end(self, outputs):
        """Called at the end of validation epoch.
           See pytorch-lightning for functionality
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.val_loss = avg_loss.item()
        logits = torch.cat([x['logits'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])
        metrics = {"val_loss": avg_loss}
        print('')
        print('avg loss:', self.val_loss)

        if self.trainer.use_ddp:
            logits_all = self.sync_across_gpus(logits)
            targets_all = self.sync_across_gpus(targets)
            predictions_all = logits_all.cpu().detach().numpy()
            targets_all = targets_all.cpu().detach().numpy()
            pearsonr_correlation = pearsonr(targets_all, predictions_all)[0]
            #print(list(zip(targets_all, predictions_all)))
            print('pearson correlation (all)', pearsonr_correlation)
            print('spearman rho (all)', spearmanr(targets_all, predictions_all)[0])
            print('loss (all)', ((targets_all - predictions_all)**2).mean())
            self.targets_all = targets_all
            self.predictions_all = predictions_all 

        if self.trainer.global_step > 0:
            self.logger.experiment.add_scalar("Val/epoch/loss", avg_loss, self.current_epoch)
            #self.logger.experiment.add_scalar("Val/epoch/pearson", avg_pearson, self.current_epoch)

        return metrics

    def test_step(self, batch, batch_idx):
        """Test step for model

        Args:
            batch: A batch of samples to train on
            batch_idx: The index of the current batch
        """
        x = batch["input_ids"]
        mask = batch["input_masks"]
        y = batch["targets"]

        outputs = self.model(x,mask,y)
        if isinstance(outputs[0],tuple):
            loss, metrics = outputs[0]
        else:
            loss = outputs[0]
            metrics = {}

        return {'test_loss': loss, 'logits': outputs[1].squeeze(1), 'targets': y.squeeze(1)}

    def test_epoch_end(self, outputs):
        """Called at the end of test epoch.
           See pytorch-lightning for functionality
        """
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.test_loss = avg_loss.item()
        logits = torch.cat([x['logits'] for x in outputs])
        targets = torch.cat([x['targets'] for x in outputs])
        metrics = {"test_loss": avg_loss}
        self.predictions = logits
        self.targets = targets

        if self.trainer.use_ddp:
            logits_all = self.sync_across_gpus(logits)
            targets_all = self.sync_across_gpus(targets)
            predictions_all = logits_all.cpu().detach().numpy()
            targets_all = targets_all.cpu().detach().numpy()
            pearsonr_correlation = pearsonr(targets_all, predictions_all)[0]
            print('pearson correlation (all)', pearsonr_correlation)
            print('spearman rho (all)', spearmanr(targets_all, predictions_all)[0])
            print('loss (all)', ((targets_all - predictions_all)**2).mean())
            self.predictions_all = predictions_all
            self.targets_all = targets_all
        return metrics

    def configure_optimizers(self):
        """Setup optimizer"""
        optimizer = utils.setup_optimizer(self.model, self.lr)
        return optimizer

    # learning rate warm-up
    """
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False,
        using_native_amp=False, using_lbfgs=False):
        
        #Customized optimizer step to allow warmup scheduling.
        #See pytorch-lightning for details
        
        # warm up lr
        lr_scale = 1
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

        self.logger.experiment.add_scalar("lr/lr_scale", lr_scale, self.trainer.global_step)
        self.logger.experiment.add_scalar("lr/lr", lr_scale*self.lr, self.trainer.global_step)

        # update params
        optimizer.step()
        optimizer.zero_grad()
    """