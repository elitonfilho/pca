This repository contains the code for the paper "Multispectral image segmentation with dimensionality reduction using task-driven auto encoders"

There are 4 main python notebooks used in the research:
- train.ipynb: Trains either autoencoder / segmentation models / autoencoder + segmentation modules
- test.ipynb Runs inference for segmentation and autoencoder models
- recoverPCA.ipynb: Applies PCA on LandCover dataset and stores it in a .npy format
- generateMetrics.ipynb: Evaluation metrics for the segmentation maps

The experiments were run using LandSat8 bands in the Chesapeake Landcover Dataset (https://lila.science/datasets/chesapeakelandcover)