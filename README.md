# biotransfer: A repository for designing sub-nanomolar antibodies using machine learning-driven approach

Please refer to our paper [Machine Learning Optimization of Candidate Antibodies Yields Highly Diverse Sub-nanomolar Affinity Antibody Libraries](https://www.biorxiv.org/content/10.1101/2022.10.07.502662v1) for additional information on the method and design of scFv sequences.  

The initial AlphaSeq Antibody Dataset 1 can be found [here](https://github.com/mit-ll/AlphaSeq_Antibody_Dataset).
Additional information about the design of Dataset 1 and experimental set-up for quantitative binding
measurements can be found in our [Data Descriptor
Paper](https://www.nature.com/articles/s41597-022-01779-4). 
The validation dataset can be found here. 

## Overview
Therapeutic antibodies are an important and rapidly growing drug modality. However, the design and discovery of early-stage antibody therapeutics remain a time and cost-intensive endeavor. Machine learning has demonstrated potential in accelerating drug discovery. We implement an Bayesian, language model-based method for desiging large and diverse libraries of target-specific high-affinity scFvs. 

The software includes four major components: 
- Large-scale antibody and protein language model training
- Functional property prediction leveraging pretrained language models to construct a probablistic antibody fitness function 
- ScFv optimization and design
- Results analysis

<img src="https://github.com/AIforGreatGood/biotransfer/blob/main/images/antibody_edited.png" width="600" height="430"> 

## System Requirement
### Hardware requirements
The package requires access to GPUs

### Software requirements
OS Requirements: This package is supported for Linux. The package has been tested on Ubuntu 18.04 and GridOS 18.04.6 

Python Dependencies: All dependencies are listed in the environment.yaml file

## Installation Guide 

(1) Install the virtual environment by running "conda env create -f environment.yaml". The one-time installation of the virtual environment takes about 10-15 mins.

(2) Switch to the new conda environment with "conda activate <env_name>". Here the env_name is "antibody_design" as defined in environment.yaml

## Demo
### Demo 1: To train a scFv binding prediction model (requires GPU)
(1) Copy & paste the pretrained language model "pfam.ckpt" into the "src/pretrained_models" directory

(2) Edit the config file "configs/lm_gp_configs/train_exact_gp_pca_14H.yaml" by replacing the \<Full Path\> with the actual path to the biotransfer folder

(3) Run the training code on a GPU node by running the following on the command line, "python train_GP_from_config.py -cd configs/lm_gp_configs/ -cn train_exact_gp_pca_14H.yaml"

Training time: Around 15 mins.

Expected output: 
- Once the model finishes training, it'll display the performance of the model on the validation data. Expect the output with MAE to be around 0.45 and Pearson correlation to be around 0.6.
- The program will automatically save 3 files in the results folder to be used for antibody generation: GP_model_state_dict.pth, pca_model.sav adn train_GP_from_config.log

### Demo 2: To generate scFvs using the trained binding (requires GPU)
(1) Edit the config file "configs/design_configs/design_pipeline_gp_14H_hc.yaml" by replacing the \<Full Path\> with the actual path to the pretrained models and training data.

(2) Run the antibody generation code by running the following on the command line "python run_design_from_config.py -cd configs/design_configs/ -cn design_pipeline_gp_14H_hc.yaml"

Expected output:
- As the program loads the model and runs the sampling, at the end of each round, the program will output the current best sequence with the predicted mean and standard deviation.  
- A list of best sequences and their corresponding score will be save in the results folder (e.g., hillclimb.csv in this case)

Time: Each round takes about 22s and the number of rounds depends (typically around 10 or so). 

### Demo 3: Experimental Validation (No GPU is required)
Two notebooks are provided in the notebooks folder demonstrating the experimental data analysis of the designed scFv sequences. The notebooks can be ran in any Python 3.8 and above environment.

## Intructions for use
See the Demo section for detailed instruction on generating artge-specific heavy-chain antibodies using the pretrained pfam language model, Gaussian process and Hill Climb sampling algorithm. The instruction below includes steps for re-creating all the models and results presented in the paper.

### 1. Language Model Training and Evaluation
- To train interactively, use "python train_from_config.py -cd configs/language_modeling_configs/ -cn \<config file here\>".  
- To train in multirun, use "python train_from_config.py -cd configs/language_modeling_configs/ -cn \<config file here\> -m".  
- To evaluate, use "python eval_from_config.py -cd configs/language_modeling_configs/ -cn \<config file here\>".  

The training of language models leverages large-scale protein and antibody sequence databases (such as Pfam for proteins and OSA for antibodies) to capture structural, evolutionary and functional properties across protein and antibody spaces. These models, onces trained, do not need to be re-trained and can be directly used to train models for predicting the downstream functional property (such as scFv binding prediction) via transfer learning. 

### 2. Training and Evaluating a Regression Model (e.g., binding prediction) 
#### Using Pretrained Language Model Finetuning
- "python train_from_config.py -cd configs/lm_finetune_configs/ -cn \<config file here\>".
- "python eval_from_config.py -cd configs/lm_finetune_configs/ -cn \<config file here\>"

#### Using Gaussian Process
- To run interactively, use "python train_GP_from_config.py -cd configs/lm_gp_configs/ -cn \<config file here\>".  
- To run interactively, use "python eval_GP_from_config.py -cd configs/lm_gp_configs/ -cn \<config file here\>".  

### 3. Design Antibody Sequences
- "python run_design_from_config.py -cd configs/design_configs/ -cn \<config file here\>"

### 4. Experimental Data Analysis (For reproducing key figures in the paper)
See jupter notebooks for detailed analysis and demonstration


## Citation Guidelines

Methodology https://www.biorxiv.org/content/10.1101/2022.10.07.502662v1
```
@article{li2022machine,
  title={Machine Learning Optimization of Candidate Antibodies Yields Highly Diverse Sub-nanomolar Affinity Antibody Libraries},
  author={Li, Lin and Gupta, Esther and Spaeth, John and Shing, Leslie and Jaimes, Rafael and Caceres, Rajmonda Sulo and Bepler, Tristan and Walsh, Matthew E},
  journal={bioRxiv},
  pages={2022--10},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```

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

## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

© 2022 Massachusetts Institute of Technology.

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

