# Analysis and data repository for Geiller, Vancura, et al., Neuron (2020)

This repository contains code and links to dataset from Geiller T., Vancura B., et al. Neuron (2020).


### Analysis code for machine-learning classification of interneuron subtypes:

The main script of the code is **CNN_Classification**.

#### Prerequisites

* [Python](http://python.org) 3.5 
* [numpy](http://www.scipy.org) >= 1.18.1
* [scipy](http://www.scipy.org) >= 1.4.1
* [scikit-image](http://scikit-image.org) >= 0.11.0
* [scikit-learn](http://scikit-learn.org) >= 0.22.1
* [tensorflow](https://www.tensorflow.org) == 2.1.0
* [keras](https://keras.io) == 2.3.1
* [seaborn](https://seaborn.pydata.org/) >= 0.10.1
* [matplotlib](https://matplotlib.org/) >= 3.1.3
* [sympy](https://www.sympy.org) >= 1.5.1

#### Description
It returns the training and testing confusion matrices as well as the training and testing accuracy performance for each cell separately.

The code is formulated to classify the following subtypes:
```
3-class problem (AAC-BC, BISTR-SOM, CCK)
4-class problem (AAC-BC, BISTR-SOM, CCK, NPY)
5-class problem (AAC, BC, BISTR, CCK, SOM), and
6-class problem (AAC, BC, BISTR, SOM, CCK, NPY)
```
The following parameters are defined by the user:

* _num_iters_ --- Number of iterations for the random train-test splits
* _epochs_ --- Number of epochs required for the CNN model to be trained
* _loss_func_ --- Loss function for the optimizer
* _learning_rate_ --- Learning rate parameter for the optimizer
* _features_ --- Number of features to feed the CNN model. The options for this parameter are the following:
  * _1_: 1D-CNN is fed with calcium signal laps
  * _2_: 2D-CNN is fed with calcium-velocity signal laps
  * _3_: 2D-CNN is fed with calcium-velocity-zdepth signal laps
* _indexing_ --- With this parameter the user defines the cell-types (either merged categories or separate cells) that will be used in the classification procedure 
* _balance_ --- This parameter defines the number of the training examples for the merged categories. The options for this parameter are the following: 
  * _balanced_: Same number of training examples for each category
  * _stratified_: It preserves the proportion of target as in original dataset, in the train datasets.
* _category_ --- The options for this parameter are the following
  * _min_categ_: As a number of training examples for each category we use the number of examples from the category with the minimum number of examples. 
  * _imbalanced_: For each category we use all its available examples for training
  * _semi_balanced_: All categories start with the minimum number of training examples (min_categ) and the user adds extra examples via the parameter size_increased, so that for each category almost the same number of training examples are used.

* _size_increased_ --- If category parameter is defined as semi_balanced, in size_increased the user defines how many examples should be added to each category. 
* _test_size_ --- User defines the number of testing examples to be used from each category
* _step_laps_ --- Number of laps to be merged
* _interp_timesteps_ --- Length of the interpolated signal laps

#### Authors
eirinitroul [AT] gmail [DOT] com; chavlis [DOT] spiros [AT] gmail [DOT] com

### Access to the dataset:

The dataset will be made available soon. 

