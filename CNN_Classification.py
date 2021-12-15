# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:47:34 2020

@author: troullinou
"""
#------------------------------------------------------------------------------------------------
import pickle
from functions import CNN_classifier, predictions

# Load the Data
with open('data/calcium_signal_df_f_all.pkl', 'rb') as file:
    df_f = pickle.load(file)

with open('data/velocity_all.pkl', 'rb') as file:
    vel = pickle.load(file)

with open('data/depth_all.pkl', 'rb') as file:
    dep = pickle.load(file)

with open('data/position_all.pkl', 'rb') as file:
    pos = pickle.load(file)
    
with open('data/labels_all.pkl', 'rb') as file:
    labels = pickle.load(file)

# DATASET PARAMETERS
sample = 'semi_balanced'  # options: 'min_categ', 'semi_balanced', 'imbalanced'
balance = 'equal'  # options: 'equal', 'stratified' (number of training examples for the merged categories)

categs = [['BC','AAC'], ['SOM', 'BISTR'], ['CCK']] # neuron cell-types selection  
test_size = [[100, 100], [100, 100], [100]]  # test set size
size = [1000, 1000, 0]  # number of extra examples for each category when the category parameter is defined as 'semi-balanced'        

# MODEL PARAMETERS
num_iters = 5  # number of random train-test splits
epochs = 100  # number of epochs          

results, model =  CNN_classifier(calcium_df_f=df_f, position=pos, labels=labels,
                                  categories=categs, test_size=test_size,
                                  sampling=sample, balance=balance, epochs=epochs,
                                  number_of_iterations=num_iters, size_increased=size,
                                  velocity=vel, depth=dep, plot=True)



# to make new predictions on new data
# preds = predictions(calcium_df_f=df_f, new_df_f=df_f, position=pos, labels=labels,
#                     categories=categs, test_size=test_size,
#                     sampling=sample, balance=balance, epochs=epochs,
#                     number_of_iterations=num_iters, size_increased=size,
#                     velocity=vel, depth=dep, plot=False,
#                     new_velocity=vel, new_depth=dep)