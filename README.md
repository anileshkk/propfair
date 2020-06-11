# propfair
Fair for All: Proportional Fairness in Classification

This repository provides code to accompany the above-mentioned paper. The code for the COMPAS and Adult data sets are in separate folders. Each folder contains the corresponding data sets, and the following files:
1. preprocess.py -- for pre-processing the data and generating train and test sets.
2. ordered_subsets.py -- which recreates the plots involving subsets obtained by ordering the data points in the test set according to the confidence given by different classifiers.
3. fixed_size.py -- which takes a fractional value (size of the subset) as input and recreates the plots which vary the composition of the subset in question based on the output of the different classifiers.
