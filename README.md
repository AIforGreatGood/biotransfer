# biotransfer

## Overview
Methods for learning biological sequence representations from large, general purpose protein datasets have demonstrated impressive ability to capture structural and functional properties of proteins. However, existing works have not yet investigated the value, limitations and opportunities of these methods in application to antibody-based drug discovery. This work provides necessary tools, models and data for comparing three classes of models: conventional statistical sequence models, supervised learning on each dataset independently, and fine-tuning an antibody specific pre-trained embedding model. 

## Setup
(1) Install the virtual environment by running "conda env create -f environment.yaml".  

(2) Switch to the new conda environment with "conda activate <env_name>". Here the env_name is "biotransfer" as defined in environment.yaml

(3) Create the config file that will run your application, and make sure the filesystem has access to the data and dataset/model source code. Use existing config files/datasets/models for guidance on how to build a new one. 

## Training a Language Model
- To run interactively, use "python train_from_config.py --config-path=\<config path here\>".  
- To run in multirun, use "python train_from_config.py --config-path=\<config path here\> -m".  
  
## Evaluating a Language model
- To run interactively, use "python eval_from_config.py --config-path=\<config path here\>".  
- To run in multirun, use "python eval_from_config.py --config-path=\<config path here\> -m". 

## Training a Downstream Model with Gaussian Process
- To run interactively, use "python train_GP.py --config-path=\<config path here\>".  
- To run in multirun, use "python train_GP.py --config-path=\<config path here\> -m". 

## Evaluating a Downstream Model with Gaussian Process
- To run interactively, use "python eval_GP.py --config-path=\<config path here\>".  
- To run in multirun, use "python eval_GP.py --config-path=\<config path here\> -m". 

## Citation Guidelines

Pfam (Pretraining) https://github.com/songlab-cal/tape#lmdb-data
```
@article{pfam,
author = {El-Gebali, Sara and Mistry, Jaina and Bateman, Alex and Eddy, Sean R and Luciani, Aur{\'{e}}lien and Potter, Simon C and Qureshi, Matloob and Richardson, Lorna J and Salazar, Gustavo A and Smart, Alfredo and Sonnhammer, Erik L L and Hirsh, Layla and Paladin, Lisanna and Piovesan, Damiano and Tosatto, Silvio C E and Finn, Robert D},
doi = {10.1093/nar/gky995},
file = {::},
issn = {0305-1048},
journal = {Nucleic Acids Research},
keywords = {community,protein domains,tandem repeat sequences},
number = {D1},
pages = {D427--D432},
publisher = {Narnia},
title = {{The Pfam protein families database in 2019}},
url = {https://academic.oup.com/nar/article/47/D1/D427/5144153},
volume = {47},
year = {2019}
}
```

OSA (Pretraining): http://opig.stats.ox.ac.uk/webapps/oas/oas_paired, http://opig.stats.ox.ac.uk/webapps/oas/oas
```
article{kovaltsuk2018observed,
  title={Observed antibody space: a resource for data mining next-generation sequencing of antibody repertoires},
  author={Kovaltsuk, Aleksandr and Leem, Jinwoo and Kelm, Sebastian and Snowden, James and Deane, Charlotte M and Krawczyk, Konrad},
  journal={The Journal of Immunology},
  volume={201},
  number={8},
  pages={2502--2509},
  year={2018},
  publisher={Am Assoc Immnol}
```

LL-Sars-Cov2 Antibody data https://github.com/mit-ll/AlphaSeq_Antibody_Dataset
```
@dataset{matthew_walsh_2021_5095284,
  author       = {Matthew Walsh and Leslie Shing and Joshua Dettman and Darrell Ricke and David Younger and Randolph Lopez and
                  Emily Engelhart and Ryan Emerson and Charles Lin and Mary Kelley and Daniel Guion},
  title        = {{mit-ll/AlphaSeq\_Antibody\_Dataset: Initial release of AlphaSeq Antibody Dataset}},
  month        = jul,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.5095284},
  url          = {https://doi.org/10.5281/zenodo.5095284}
}
```

IgG Antibody data https://github.com/gifford-lab/antibody-2019/tree/master/data/training%20data
```
article{liu2020antibody,
  title={Antibody complementarity determining region design using high-capacity machine learning},
  author={Liu, Ge and Zeng, Haoyang and Mueller, Jonas and Carter, Brandon and Wang, Ziheng and Schilz, Jonas and Horny, Geraldine and Birnbaum, Michael E and Ewert, Stefan and Gifford, David K},
  journal={Bioinformatics},
  volume={36},
  number={7},
  pages={2126--2133},
  year={2020},
  publisher={Oxford University Press}
```

## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

© 2021 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
SPDX-License-Identifier: MIT

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

The software/firmware is provided to you on an As-Is basis
