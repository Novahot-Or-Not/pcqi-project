# Machine Learning Neutrino Signatures
A project for Physics of Classical/Quantum Information.

This project  aims at classifying neutrino track and shower signatures by using Machine Learning (ML) methods (SVC : support vector classifier, RFC : random forest classifier). The used data has been provided by Dr. Dorothea Samtleben and originates from a simulation representing ten detector units corresponding to the detector strings of the KM3NeT project (see [KM3NeT website](https://www.km3net.org/). Furthermore, it contains parameters from a fit of the data and likelihoods for track and shower start. In the scope of this project, the following steps have been of importance :

## Input data
Note : Our functions expect .h5 files as input data and our models are stored in .parquet files.

## Pre-analysis of the data
Looking at properties of our data without doing any ML.
### Distribution of track start
In plot_density.py there is a density plot of the track start variable (Track x-position, Track y-position) to see where the pre-made fit has put the track starts and whether it makes sense with the simulated environment as background.
### Track and shower likelihood
In LikelihoodAnalysis.py one wants to know, whether it is possible to visually separate track and shower events when plotting data of the two signatures. The parameters of likelihood for a track and the likelihood for a shower are plotted against each other, once in scatter plots and also with 2d histograms to see the amount of events on each point on the graphs. Besides, this is done for high-energy events, so a cutoff has to be defined. It is possible to choose between getting completely separated figures or to plot multiple graphs onto one figure.

## Training model
### utils.py
This file contains functions used by the other files, as well as a list, used_columns, which contains the names of the features to be used in training.
### train.py
This is the main file, where the following things are being done :
1. Loading data : load from .h5 file in folder named "data", rename columns and  add columns "Is shower?" and "Particle name"
2. Pre-processing data :
	- exclude unused columns, remove rows with missing entries, balance data to get same amount of each particle type, normalise data and save to .parquet file
	- if pre-processed data file already exists : load dataframe from file
3. Splitting data : divide data into training and validation set, each containing balanced amount of particle type events
4. Training model : train model with classifier of choice (comment out other classifiers) and by setting parameters of choice (this is done at the top of the code)
	- Linear SVC : set "dual" parameter
	- SVC : set "kernel" parameter
	- RFC : set number of trees "n_estimators"
5. Validating model : determine accuracy of model by using validation set
6. Saving model :Save model data to .parquet file in folder named "models"

## Analysis of SVC model coefficients
analyse_model_coefficients.py
Analyse feature importance for the linear SVC models by loading corresponding .parquet file and by plotting a bar plot.

## Predicting signatures using a model
(not implemented yet)
