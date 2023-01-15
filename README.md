# What Can the Neural Tangent Kernel Tell Us About Adversarial Robustness?

This repository contains source code for the main experiments of the NeurIPS 2022 paper titled 'What Can the Neural Tangent Kernel Tell Us About Adversarial Robustness?'. 

## Paper experiments
`robust_train_CNN.py` contains code to adversarially train a neural network and compute (& save) empirical ntks during training.
`measurements.py` contains code for the computation of several kernel quantities on precomputed ntks.

## Notebook demo
`NTK_features_example.ipynb` is a self-contained notebook (that can be run on google colab) that demonstrates NTK features, as defined in the paper.
