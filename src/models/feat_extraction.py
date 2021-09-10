# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from pytorch_lightning import LightningModule
import torch
from tape.models.modeling_utils import ProteinConfig
from .modeling_bert import BertFeatureExtractor, BertTokenFeatureExtractor
from .CNN import CNN

class BertFeatExtractor(LightningModule):
    """Feature extractor based on Bert"""

    def __init__(self, model_config_file):
        """Inits model.

        Args:
            model_config_file: Path to config file
        """
        super().__init__()
        config = ProteinConfig().from_pretrained(model_config_file)
        self.model = BertFeatureExtractor(config)
    
    def forward(self, input_ids, input_mask=None, targets=None, variable_regions=None, feat_concat=False):
        """Forward hook for model"""
        return self.model(input_ids, input_mask=input_mask, targets=targets, variable_regions=variable_regions, feat_concat=feat_concat)

class BertTokenEmbedding(LightningModule):
    """Feature extractor based on Bert"""

    def __init__(self, model_config_file, model_checkpoint_path, strict_reload=True):
        """Inits model.

        Args:
            model_config_file: Path to config file
        """
        super().__init__()
        config = ProteinConfig().from_pretrained(model_config_file)
        self.model = BertTokenFeatureExtractor(config)

        state_dict = {}
        for k,v in torch.load(model_checkpoint_path)['state_dict'].items():
            state_dict['.'.join(k.split('.')[1:])] = v
        self.model.load_state_dict(state_dict, strict=strict_reload)
    
    def forward(self, input_ids, input_mask=None):
        """Forward hook for model"""
        return self.model(input_ids, input_mask=input_mask)

class ConvNetFeatExtractor(CNN):
    """Feature extractor based on Bert"""

    def __init__(self, enc_channels, enc_kernel_sizes, enc_strides, enc_paddings, enc_hidden_sizes, mp_kernel_size, mp_stride, dropout=0):
        super().__init__(enc_channels, enc_kernel_sizes, enc_strides, enc_paddings, enc_hidden_sizes, mp_kernel_size, mp_stride, dropout=0)
        
    def forward(self, x):
        """Forward hook for model"""
        latent, conv_shape = self.encode(x)
        return latent
