# Gated Additive Tree Ensemble(GATE)

We propose a novel high-performance, parameter and computationally efficient deep learning architecture for tabular data, Gated Additive Tree Ensemble(GATE). GATE uses a gating mechanism, inspired from GRU, as a feature representation learning unit with an in-built feature selection mechanism. We combine it with an ensemble of differentiable, non-linear decision trees, re-weighted with simple self-attention to predict our desired output. We demonstrate that GATE is a competitive alternative to SOTA approaches like GBDTs, NODE, FT Transformers, etc. by experiments on several public datasets (both classification and regression). We have released the code under the MIT license. [arxiv Paper]()

## Installation

Although the PyTorch Tabular (which is a dependency) installation includes PyTorch, the best and recommended way is to first install PyTorch from [here](https://pytorch.org/get-started/locally/), picking up the right CUDA version for your machine.

Once, you have got Pytorch installed, clone the repository and run the following command:
```
 pip install -r requirements.txt
```

to install the all the dependencies.


## Usage

`main.py` is the main file of the GATE package. It is a python script that can be used to train GATE models and replicate the r=experiments.

The file can be used as a command line tool as well as programatically to train GATE models. 
`run` function inside `main.py` takes the following arguments:

```
usage: main.py [-h] [-cft none] [-ncf True] [-lr 0.001] [-alr False] [-bs 1024] [-e 100] [-es True] [-gpu True] [-o adam] [-wd 1e-05] [-lrs cosine] [-fau 6] [-fadp 0.0] [-nt 20] [-td 6] [-twa True] [-twd 0.0] [-fmf entmax] [-ba entmoid] [-tdr 0.0] [-shw False]
               [-ctree True] [-bnci False] [-drop 0.0] [-te False]
               [{FOREST,CLICK,MICROSOFT,YEAR,A9A}] [checkpoints_path] [experiment_name]

positional arguments:
  {FOREST,CLICK,MICROSOFT,YEAR,A9A}
                        Dataset to use
  checkpoints_path      [saved_checkpoints/]
  experiment_name       <class 'str'>

options:
  -h, --help            show this help message and exit
  -cft none, --continuous-feature-transform none
                        Continuous feature transform to use
  -ncf True, --normalize-continuous-features True
                        Normalize continuous features
  -lr 0.001, --learning-rate 0.001
                        Learning rate
  -alr False, --auto-lr-find False
                        Auto lr find
  -bs 1024, --batch-size 1024
                        Batch size
  -e 100, --max-epochs 100
                        Max epochs
  -es valid_loss, --early-stopping valid_loss
                        Early stopping. The metric to monitor for early stopping
  -gpu True, --use-gpu True
                        Use GPU
  -o adam, --optimizer adam
                        Optimizer
  -wd 1e-05, --weight-decay 1e-05
                        Weight decay
  -lrs cosine, --learning-rate-scheduler cosine
                        LR scheduler
  -fau 6, --gflu-stages 6
                        Feature abstraction stages
  -fadp 0.0, --gflu-dropout 0.0
                        Feature abstraction dropout
  -nt 20, --num-trees 20
                        Num trees
  -td 6, --tree-depth 6
                        Depth of Trees
  -twa True, --tree-wise-attention True
                        Tree wise attention
  -twd 0.0, --tree-wise-attention-dropout 0.0
                        Dropout for Tree wise attention
  -fmf entmax, --feature-mask-function entmax
                        Feature mask function
  -ba entmoid, --binning-activation entmoid
                        Binning activation
  -tdr 0.0, --tree-dropout 0.0
                        Tree dropout
  -shw False, --share-head-weights False
                        Share head weights
  -ctree True, --chain-trees True
                        Chain trees
  -bnci False, --batch-norm-continuous-input False
                        Batch norm continuous
  -drop 0.0, --dropout 0.0
                        Dropout
  -te False, --track-experiment False
                        Track experiment
```

## Files

The code for the GATE model is inside `gate` folder. GATE uses PyTorch Tabular to load the data and train the model.

```
-gate
  -  attention_forest.py - Has all the core code for the GATE model.
  -  config.py - Has all the configurations for the GATE model.
  -  utils.py - Has all the utility functions for the GATE model.
```