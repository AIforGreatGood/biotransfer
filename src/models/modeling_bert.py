# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

# This file is a modification of https://github.com/songlab-cal/tape, release version 0.4. 
# MIT Lincoln Laboratory has made these modifications to add support to expose hidden
# hyperparameters/variables, such as position_ids, token_type_ids, hid_dim, dropout,
# for training a paired antibody language model, and enabling hyperparameter optimization
# and more flexible feature extraction. 

# Copyright (c) 2018, Regents of the University of California
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Functions that expose hidden hyperparameters/variables that were not exposed by TAPE

Exposed hyperparameters/variables are
    ProteinBertForMaskedLM: token_type_ids to enable paired protein/antibody training
    ProteinBertForValuePrediction: hidden_size and dropout rate for the downstream value prediction task to enable hyperparameter sweep
"""
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import torch
from torch import nn
from tape.models.modeling_bert import ProteinBertAbstractModel, ProteinBertEmbeddings, ProteinBertEncoder, ProteinBertPooler
from tape.models.modeling_utils import ProteinModel, MLMHead, accuracy
from torch.nn.utils.weight_norm import weight_norm
from typing import List
from collections import defaultdict

# ------------------------ ProteinBertForMaskedLM -----------------------------#
class ProteinBertModel(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = ProteinBertEmbeddings(config)
        self.encoder = ProteinBertEncoder(config)
        self.pooler = ProteinBertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class ProteinModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self,
                input_ids,
                input_mask=None,
                position_ids=None,
                token_type_ids=None):
        if input_mask is None:
            input_mask = torch.ones_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)

        # Since input_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       chunks=None)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class ProteinBertForMaskedLM(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBertModel(config)
        self.mlm = MLMHead(
            config.hidden_size, config.vocab_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.mlm.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self,
                input_ids,
                input_mask=None,
                targets=None,
                position_ids=None,
                token_type_ids=None):

        outputs = self.bert(input_ids, input_mask=input_mask, position_ids=position_ids, token_type_ids=token_type_ids)

        sequence_output, pooled_output = outputs[:2]
        # add hidden states and attention if they are here
        outputs = self.mlm(sequence_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs

        

# ------------------------ ProteinBertForValuePrediction -----------------------------#
class SimpleMLP(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None))

    def forward(self, x):
        return self.main(x)

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        eps = 1e-6
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y) + eps)
        return loss

class ValuePredictionHead(nn.Module):
    def __init__(self, hidden_size_bert: int, hidden_size, dropout: float = 0., loss_function='mse'):
        super().__init__()
        self.value_prediction = SimpleMLP(hidden_size_bert, hidden_size, 1, dropout)
        self.loss_function = loss_function

    def forward(self, pooled_output, targets=None):
        value_pred = self.value_prediction(pooled_output)
        outputs = (value_pred,)

        if targets is not None:
            if self.loss_function == 'mse':
                loss_fct = nn.MSELoss()
            elif self.loss_function == 'l1':
                loss_fct = nn.L1Loss()
            elif self.loss_function == 'rmse':
                loss_fct = RMSELoss()
            else:
                raise ValueError('Unknown loss function') 
            value_pred_loss = loss_fct(value_pred, targets)
            outputs = (value_pred_loss,) + outputs 
        return outputs  # (loss), value_prediction

class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size_bert: int, hidden_size, num_labels, dropout: float = 0.):
        super().__init__()
        self.classify = SimpleMLP(hidden_size_bert, hidden_size, 1, dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pooled_output, targets=None, weights=None):
        logits = self.classify(pooled_output)
        logits = self.sigmoid(logits).squeeze()
        outputs = (logits,)

        if targets is not None:
            """
            if weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            """
            loss_fct = nn.BCELoss()
            classification_loss = loss_fct(logits, targets)
            #metrics = {'accuracy': accuracy(logits, targets)}
            #loss_and_metrics = (classification_loss, metrics)
            loss_and_metrics = classification_loss
            outputs = (loss_and_metrics,) + outputs

        return outputs  # (loss), logits

class ProteinBertForValuePrediction(ProteinBertAbstractModel):

    def __init__(self, config, hidden_size: int, dropout: float = 0., loss_function='mse'):
        super().__init__(config)
        
        self.bert = ProteinBertModel(config)
        self.predict = ValuePredictionHead(config.hidden_size, hidden_size, dropout, loss_function)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)
        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs

class ProteinBertForSequenceClassification(ProteinBertAbstractModel):

    def __init__(self, config, hidden_size: int, num_labels: int, dropout: float = 0.):
        super().__init__(config)

        self.bert = ProteinBertModel(config)
        self.classify = SequenceClassificationHead(
            config.hidden_size, hidden_size, num_labels, dropout)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None, weights=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]

        outputs = self.classify(pooled_output, targets, weights) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs


class BertFeatureExtractor(ProteinBertAbstractModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = ProteinBertModel(config)
        self.hidden_size_bert = config.hidden_size

    def forward(self, input_ids, input_mask=None, targets=None, variable_regions:List[int]=None, feat_concat:bool=False):
        outputs = self.bert(input_ids, input_mask=input_mask)
        #sequence_output = outputs[0]
        sequence_output = outputs[2][-2]    
        #sequence_output = torch.cat((sequence_output[-1], sequence_output[-2]), dim=2) # concat the last 4 layers 
        if variable_regions is not None: # extract the variable regions
            if not isinstance(variable_regions[0], list): # not a list of lists
                variable_regions = [variable_regions]
            sequence_output = [sequence_output[:,v] for v in variable_regions]
            pooled_output = torch.cat(sequence_output, dim=1) 
            if feat_concat:
                pooled_output = pooled_output.view(pooled_output.size(0),-1)
            else:
                pooled_output = torch.mean(pooled_output, dim=1)
        else:
            assert not feat_concat, 'If no variable regions is specified, then features cannot be concatenated. Either set variable_regions=True or set feat_concat=False'
            # convert the first and last masked token into 0
            input_mask[:,0]=0
            tmp_dict = defaultdict(list)
            for i,j in torch.nonzero(input_mask):
                tmp_dict[int(i)].append(int(j))
            last_token_ind = []
            for k,val in tmp_dict.items():
                input_mask[k,max(val)] = 0
            mask = input_mask.unsqueeze(-1).expand(sequence_output.size())
            sequence_output = sequence_output*mask
            pooled_output = torch.sum(sequence_output, dim=1)
            scale = 1/torch.sum(input_mask, dim=1)
            scale = scale.unsqueeze(-1).expand(pooled_output.size())
            pooled_output = pooled_output*scale
            """
            pooled_output = torch.mean(sequence_output, dim=1)
            """
        return pooled_output

class BertTokenFeatureExtractor(ProteinBertAbstractModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = ProteinBertModel(config)

    def forward(self, input_ids, input_mask=None):
        outputs = self.bert(input_ids, input_mask=input_mask)
        sequence_output = outputs[0]
        return sequence_output

