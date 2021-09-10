# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""Helper class for evaluating cov2 data"""

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn import metrics
#from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt  
from pytorch_lightning.utilities import rank_zero_only

class sars_cov2_eval():
    """Helper class for evaluating cov2 data"""

    # seed sequence value (extracted from the controls)
    seed_sequence_value = {
        "14H": 1.3445717628028468,
        "14L": 1.3445717628028468,
        "91H": 1.88361976098417,
        "95L": 2.3378946100102533,
    }

    # cuttoff value (extracted by fitting a bimodal model and compute the intersection)
    cutoff_value = {
        "14H": 4,
        "14L": 3.8,
        "91H": 3.95,
        "95L": 3.5,
    }

    def __init__(self, chain: str):
        """Inits class

        Args:
            chain: Chain of covid antibody
        """
        self.chain = chain
        self.cutoff_0 = self.__class__.seed_sequence_value[chain]
        self.cutoff_1 = self.__class__.cutoff_value[chain]

    @rank_zero_only
    def evaluate(self, targets, predictions):
        """Evaluate prediction performance

        Args:
            targets: True binding affinities for covid
            predictions: Predicted binding affinities
        """
        # pearson evaluation with respect to different cutoff values
        for cutoff in [self.cutoff_0, self.cutoff_1]:
            ind = np.where(targets<cutoff)[0]
            targets_ = targets[ind]
            predictions_ = predictions[ind]
            print(f'pearson (cutoff={cutoff:.2f}):', pearsonr(targets_, predictions_)[0])
            print(f'spearman (cutoff={cutoff:.2f}):', spearmanr(targets_, predictions_)[0])
        print(f'pearson (all):', pearsonr(targets, predictions)[0])
        print(f'spearman (all):', spearmanr(targets, predictions)[0])
        print('')
        plt.scatter(targets,predictions)
        plt.show()
        plt.savefig('/home/gridsan/LI25662/ML4Bio/lightning_distributed_copy/scatterplot.png') 

        # classification evaluation with respect to different cuttoff values
        predictions = -predictions
        targets = -targets

        target_class = np.zeros(len(targets))
        target_class[targets>=-self.cutoff_0]+=1
        precision, recall, thresholds = metrics.precision_recall_curve(target_class,predictions)
        aupr = metrics.auc(recall, precision)
        print('aupr_0', aupr)
        plt.figure()
        plt.plot(recall, precision)
        plt.savefig('/home/gridsan/LI25662/ML4Bio/lightning_distributed_copy/aupr_0.png') 
        fpr, tpr, thresholds = metrics.roc_curve(target_class,predictions) 
        auroc = metrics.auc(fpr, tpr)
        print('auroc_0', auroc)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.savefig('/home/gridsan/LI25662/ML4Bio/lightning_distributed_copy/auroc_0.png') 
        #print(list(zip(precision, recall, thresholds)))
        #print(self.cutoff_0)

        target_class = np.zeros(len(targets))
        target_class[targets>=-self.cutoff_1]+=1
        precision, recall, thresholds = metrics.precision_recall_curve(target_class, predictions)
        aupr = metrics.auc(recall, precision)
        print('aupr_1', aupr)
        plt.figure()
        plt.plot(recall, precision)
        plt.savefig('/home/gridsan/LI25662/ML4Bio/lightning_distributed_copy/aupr_1.png')  
        fpr, tpr, thresholds = metrics.roc_curve(target_class,predictions) 
        auroc = metrics.auc(fpr, tpr)
        print('auroc_1', auroc)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.savefig('/home/gridsan/LI25662/ML4Bio/lightning_distributed_copy/auroc_1.png') 





        
