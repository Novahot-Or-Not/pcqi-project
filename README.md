# Machine Learning Neutrino Signatures
A project for Physics of Classical/Quantum Information.

This project  aims at classifying neutrino track and shower signatures by using Machine Learning (ML) methods (SVC : support vector classifier, RFC : random forest classifier). The used data has been provided by Dr. Dorothea Samtleben and originates from a simulation representing ten detector units corresponding to the detector strings of the KM3NeT project (see [KM3NeT website](https://www.km3net.org/)). Furthermore, it contains parameters from a fit of the data and likelihoods for track and shower start. In the scope of this project, the following steps have been of importance :

## Input data
Note : Our functions expect .h5 files as input data and our models are stored in .joblib files.

## Pre-analysis of the data
Looking at properties of our data without doing any ML.
### Distribution of track start
In `plot_density.py` there is a density plot of the track start variable (Track x-position, Track y-position) to see where the pre-made fit has put the track starts and whether it makes sense with the simulated environment as background.
### Track and shower likelihood
In `LikelihoodAnalysis.py` one wants to know, whether it is possible to visually separate track and shower events when plotting data of the two signatures. The parameters of likelihood for a track and the likelihood for a shower are plotted against each other, once in scatter plots and also with 2d histograms to see the amount of events on each point on the graphs. Besides, this is done for high-energy events, so a cutoff has to be defined. It is possible to choose between getting completely separated figures or to plot multiple graphs onto one figure.

## How to train a model
1. In `train.py`, configure the parameters
	1. Set `filenames` to the names of the files containing the data on which you want to train the model.\
	**Note:** The code expects .h5 files within a subfolder named "data"
	2. Set `equalised_columns` to the names of the columns on which you want to balance the training data. Set `equalise_columns` to False if you do not wish to balance the data.
	3. Choose the type of model on which you wish to train. The currently available models are `LinearSVC`, `SVC`, and `RandomForestClassifier`. Comment and uncomment the relevant code in lines 77-79 to choose your model.
	4. Set the parameters specific to your chosen model.
		- For `LinearSVC`: `dual`
		- For `SVC`: `kernel`
		- For `RandomForestClassifier`: `n_estimators` and `max_depth`
	5. Set `model_filename` to the name of the model. This file will be saved in a subfolder named "models"
2. In `utils.py`, fill in names of the features which you want to use for training into `used_columns`, along with "Inelasticity", "Particle name", "Is shower?", and "is_cc".
	1. It is possible to rename some features for better readability during analysis. The function `column_renamer` contains a dictionary `rename_dict`, containing the original feature names along with their desired names. Add any features you wish to rename to this dictionary.
3. Run the script by calling `python train.py` from the command line.
4. Once the training is done, the accuracy of the model on the validation set will be outputted to the console.

**Note:** If you have previously trained a model, the preprocessed data will be stored in a .parquet file (location specified in `parquet_filepath` within `train.py`). This is to save time spent on loading and preprocessing the data. If you change the used data or any of the preprocessing steps, you must delete the .parquet file in order to see any changes. 

## Analysis of SVC model coefficients
`analyse_model_coefficients.py`
Analyse feature importance for the linear SVC models by loading corresponding .joblib file and by plotting a bar plot.

## Predicting signatures using a model
(not implemented yet)
test

## Explanation per file
### `analyse_model_coefficients.py`
Analyses the importance of each feature in the classification process and displays these in a histogram. Only available for `LinearSVC` models and `SVC` models trained with a linear kernel.

#### Instructions
1. Set `model_filename` to the name of the model.\
**Note:** The model should be in a subfolder named "models".
2. Run the script from the command line. This will open a window containing the histogram.

### `LikelihoodAnalysis.py`
Creates a scatterplot of likelihood data for track and shower.

#### Instructions
1. Set `filenames` to the names of the files containing the data to be analysed.
2. Set the energy threshold `Emin`.
3. Run the script from the command line. This will open a window containing the plot.

### `plot_density.py`
Plots the density of track reconstructions in the x-y plane and display the plot. Only uses the first `datapoint_count` particles. This is implemented as a time-saving feature.

#### Instructions
1. Set the x and y range of the window using `xmin`, `xmax`, `ymin`, and `ymax`.
2. Set the amount of datapoints to analyse using `datapoint_count`.
3. Set the grid size of the density plot using `grid_size`.
4. Run the script from the command line. This will open a window containing the density plot.

### `train.py`
Trains a machine learning model on specified data. Data is manipulated according to the following steps before training:
1. Loading data
2. Pre-processing data
3. Splitting data
4. Training model
5. Validating model
6. Saving model

Instructions on how to use can be found above.

### `utils.py`
Contains functions used by other files. Also contains a list `used_columns`, which contains the names of the features to be used in training.