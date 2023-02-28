# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

from tape import utils

from pytorch_lightning import LightningModule
import torch

from tape.models.modeling_utils import ProteinConfig
from .modeling_bert import ProteinBertForMaskedLM

class BertPairedMaskedLM(LightningModule):
    def __init__(self, model_config_file, lr, from_pretrained=None, warmup_steps=0,
        reload_weight_path=None, reload_config_path=None):

        super().__init__()
        self.save_hyperparameters()
        config = ProteinConfig().from_pretrained(model_config_file)
        self.model = ProteinBertForMaskedLM(config) 
        ### TODO: test the model loading function
        if from_pretrained is not None:
            self.model = self.model.from_pretrained(from_pretrained)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.reload_weight_path = reload_weight_path
        self.reload_config_path = reload_config_path

    def training_step(self, batch, batch_idx):
        x = batch["input_ids"]
        mask = batch["input_mask"]
        y = batch["targets"]
        position_ids = batch["position_ids"]
        token_type_ids = batch["token_type_ids"]

        outputs = self.model(x,input_mask=mask,targets=y,position_ids=position_ids,token_type_ids=token_type_ids)
        if isinstance(outputs[0],tuple):
            loss, metrics = outputs[0]
        else:
            loss = outputs[0]
            metrics = {}

        self.logger.experiment.add_scalar("Train/step/loss", loss, self.trainer.global_step)
        self.logger.experiment.add_scalar("Train/step/perplexity", metrics["perplexity"], self.trainer.global_step)
        
        #To add metrics to the progress bar, use {'progress_bar': <metric>} as a returned value. See:
        #https://github.com/PyTorchLightning/pytorch-lightning/issues/2608

        return {'loss': loss, 'progress_bar': {'Perplexity': metrics["perplexity"]}, 'metrics': metrics, "perplexity": metrics["perplexity"]}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_perplexity = torch.stack([x['perplexity'] for x in outputs]).mean()
        metrics = {"loss": avg_loss, "perplexity": avg_perplexity}

        self.logger.experiment.add_scalar("Train/epoch/loss", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Train/epoch/perplexity", avg_perplexity, self.current_epoch)

        return metrics

    def validation_step(self, batch, batch_idx):
        x = batch["input_ids"]
        mask = batch["input_mask"]
        y = batch["targets"]
        position_ids = batch["position_ids"]
        token_type_ids = batch["token_type_ids"]

        outputs = self.model(x,input_mask=mask,targets=y,position_ids=position_ids,token_type_ids=token_type_ids)
        if isinstance(outputs[0],tuple):
            loss, metrics = outputs[0]
        else:
            loss = outputs[0]
            metrics = {}

        return {'val_loss': loss, 'metrics': metrics, "val_perplexity": metrics["perplexity"]}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.val_loss = avg_loss.item()
        print("Val loss: {}".format(self.val_loss))
        avg_perplexity = torch.stack([x['val_perplexity'] for x in outputs]).mean()
        metrics = {"val_loss": avg_loss, "val_perplexity": avg_perplexity}

        if self.trainer.global_step > 0:
            self.logger.experiment.add_scalar("Val/epoch/loss", avg_loss, self.current_epoch)
            self.logger.experiment.add_scalar("Val/epoch/perplexity", avg_perplexity, self.current_epoch)

        return metrics

    def sync_across_gpus(self, t):   # t is a tensor       
        # a work-around function to sync outputs across multiple gpus to compute a metric
        gather_t_tensor = [torch.ones_like(t) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather(gather_t_tensor, t)
        return torch.cat(gather_t_tensor)

    def test_step(self, batch, batch_idx):
        x = batch["input_ids"]
        mask = batch["input_mask"]
        y = batch["targets"]
        position_ids = batch["position_ids"]
        token_type_ids = batch["token_type_ids"]

        outputs = self.model(x,input_mask=mask,targets=y,position_ids=position_ids,token_type_ids=token_type_ids)
        if isinstance(outputs[0],tuple):
            loss, metrics = outputs[0]
        else:
            loss = outputs[0]
            metrics = {}
        return {'test_loss': loss, 'metrics': metrics, "test_perplexity": metrics["perplexity"]}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.test_loss = avg_loss.item()
        perplexities = torch.stack([x['test_perplexity'] for x in outputs])
        avg_perplexity = perplexities.mean()
        metrics = {"test_loss": avg_loss, "test_perplexity": avg_perplexity}
        print('avg perplexity:', avg_perplexity)

        if self.trainer.use_ddp:
            avg_perplexity_all = self.sync_across_gpus(perplexities).mean()
        print('average perplexity (all)', avg_perplexity_all)

        return metrics

    def configure_optimizers(self):
        optimizer = utils.setup_optimizer(self.model, self.lr)
        return optimizer

    # learning rate warm-up
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False,
        using_native_amp=False, using_lbfgs=False):

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
