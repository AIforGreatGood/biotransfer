# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

from pytorch_lightning import LightningModule
import torch
from tape.models.modeling_utils import ProteinConfig
from tape.models.modeling_bert import ProteinBertAbstractModel, ProteinBertModel
from importlib import import_module

def load_featmodel_from_checkpoint(model_config_file,feat_path,strict_reload=False):
    #module = getattr(import_module("src.models.feat_extraction"), "FeatExtractor")
    #print(module)
    feat_model = FeatExtractor.load_from_checkpoint(feat_path, model_config_file=model_config_file, strict=strict_reload)
    feat_model.eval()
    return feat_model

class BertFeatureExtractor(ProteinBertAbstractModel):
    """BertFeatureExtractor as tape model"""

    def __init__(self, config):
        super().__init__(config)
        self.bert = ProteinBertModel(config)
        self.hidden_size_bert = config.hidden_size

    def forward(self, input_ids, input_mask=None, variable_regions=None,targets=None):
        outputs = self.bert(input_ids, input_mask=input_mask)
        sequence_output, pooled_output = outputs[:2]
        if variable_regions is not None: # extract the variable regions instead of the first token
            pooled_output = sequence_output[:,variable_regions].view(sequence_output.size(0),-1)
        return pooled_output

class FeatExtractor(LightningModule):
    """BertFeatureExtractor as lightning model"""
    
    def __init__(self, model_config_file):
        super().__init__()
        config = ProteinConfig().from_pretrained(model_config_file)
        self.model = BertFeatureExtractor(config)
    
    def forward(self, input_ids, input_mask=None, variable_regions=None, targets=None):
        return self.model(input_ids, input_mask=input_mask, targets=targets, variable_regions=variable_regions)
