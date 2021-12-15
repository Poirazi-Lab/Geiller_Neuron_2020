# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:24:30 2020

@author: troullinou
"""

def CNN_classifier(calcium_df_f, position, labels, categories, test_size,
                   sampling='imbalanced', balance='stratified',
                   epochs=200, number_of_iterations=10,
                   size_increased=None, velocity=None,depth=None, plot=True):

    """
    Classifier to classify the neuronal types based on several features. The 
    function saves the best model per iteration and the results, confusion matrices
    for both the training and test phase.
    'best_model_split_'+str(iters)+'.hdf5': model
    'performance.pkl': classifier's performance
    'data.pkl': data from all iterations
    
    Args:
        calcium_df_f          : (list of np.arrays) Calcium signal DeltaF/F
        position              : (list of np.arrays) animal's position on the linear track  
        labels                : (list of str) labeling from immunochemistry 
                                'SOM'   : Somatostatin positive cells
                                'AAC'   : Axoaxonic cells
                                'BC'    : Basket cells
                                'CCK'   : CCK positive cells
                                'NPY'   : neuroglia form cells
                                'BISTR' : Bistratified cells
                                'U'     : Unspecified
        categories            : (list of lists of str) categorization of cells.
                                In case of a list with more than one type, these
                                categories are merged in one super-category
        test_size             : (list of lists of int) The test size per category
                                in order for every iteration to be comparable with
                                the others.
        sampling              : (str) specify the type of sampling in the training set
                                imbalanced    : all data of each category (no subsampling)
                                semi-balanced : variable sizes (downsampling)
                                min_categ     : downsample to match the minimum sized category
        balance               : (str) In case of merged categories either use equal
                                size of each subcategory or keep the same precentage
                                as in the original dataset.Valid options: equal, stratified
        epochs                : (int) number of maximum epochs for classifier. Default is 100.
        number_of_iterations  : (int) number of random train-test splits. Default is 10.
        size_increased        : (list of int) size of each neuronal type in
                                in the training dataset above the mionimum category
                                Optional, Default is None
        velocity              : (list of floats) animal's velocity, Default None
        depth                 : (list of floats) depth of a cell, Default None
        plot                  : (boolean) True (default) for plotting, False othwerwise.    
    Returns:
        results               : (dict) Dictionary with:
                                'conf_matrix_mean': (np.array) mean confusion matrix across iteartion (categories x categories)
                                'conf_matrix_std' : (np.array) standard deviation of the confusion matrix (categories x categories)
                                'accuracy': (dict) accuracy for each iteration
                                            keys: name of the category. In case
                                            of a merge the names are separated with a dash, e.g., AAC-BC
                                'class_names': (list) of strings with categories name for plotting
    """

    import numpy as np
    import pickle, time
    
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.utils import class_weight
    from keras.utils import to_categorical
    from keras.callbacks import EarlyStopping
    from keras.callbacks import ModelCheckpoint
    from keras.models import load_model
    
    # IMPORT MY SCRIPTS/FUNCTIONS
    from functions import data_preprocessing_laps_merging_classes
    from functions import model_1D_cnn_1feature, model_2D_cnn_2features, model_2D_cnn_3features 
    from functions import custom_split, take_labels, fix_training_set, plots
    
    #------------------------------------------------------------------------------------------------ 
    # Various checks
    
    for n, item in enumerate(categories):
        if type(item) is not list:
            raise ValueError(f'The element {n+1} of categories is not a list. Please change it.')
            
    for n, item in enumerate(test_size):
        if type(item) is not list:
            raise ValueError(f'The element {n+1} test_size is not a list. Please change it.')            
            
    names = ['SOM', 'AAC', 'BC', 'CCK', 'NPY', 'BISTR', 'U']

    for nam in [item for sublist in categories for item in sublist]:
        if nam not in names:
            raise ValueError(r"Not a valid category name. Use one of 'SOM', 'AAC', 'BC', 'CCK', 'NPY', 'BISTR'.")

    L1 = len([item for sublist in categories for item in sublist])
    L2 = len([item for sublist in test_size for item in sublist])
    
    if L1!=L2:
        raise ValueError('Categories list and test_size list have different elements.')
    
    if size_increased:
       
        if len(size_increased)!=len(categories):
            raise ValueError('The size increased list do not match the size of categories.')
        
    for n in range(len(categories)):
        if len(categories[n])!=len(test_size[n]):
            raise ValueError('Categories and test_size must have the exact same structure.')

    if sampling=='semi_balanced' and not size_increased:
        raise ValueError(r"The size_increased argument is missing. Either change 'sampling' method or specify it.")

    #------------------------------------------------------------------------------------------------ 

    # for saving the results
    class_names = []
    for index in categories:
        name = ''
        if len(index)>1:
            for index2 in range(len(index)-1):
                name += index[index2]+'-'
            name += index[index2+1]
            class_names.append(name)
        else:
            class_names.append(index[0])
    
    #------------------------------------------------------------------------------------------------ 
    
    num_classes = len(categories)
    
    # GET ALL DATA
    return_list = data_preprocessing_laps_merging_classes(calcium_df_f=calcium_df_f, position=position, labels=labels,
                                                step_laps=2, interp_timesteps=100,
                                                velocity=velocity, depth=depth)
    
    data_all = return_list[0]
    labels_all = return_list[1]
    features = return_list[2]
    
    # GET ONLY THE DATA ACCORDING TO categories (line 50)
    data, labels = take_labels(data_all, labels_all, categories)
    
    input_shape = tuple(np.expand_dims(data[0], axis=-1).shape)
    
    #------------------------------------------------------------------------------------------------ 
    # INITIALIZATIONS
    train_accuracy = []
    test_accuracy = []
    time_elapsed = []
    
    # CLASS-NAMES FOR THE CONF. MATRICES
        
    train_conf_matr_all = []
    test_conf_matr_all = []
    train_accuracies_all = {}
    test_accuracies_all = {}
            
    train_all_data_list   = []
    train_all_labels_list = []
    test_all_data_list    = []
    test_all_labels_list  = []
    
    train_all_predicitions_list  = []
    test_all_predicitions_list  = []
    
    # Keep track of the best model in all iterations
    score_old = 1e-13
        
    for iters in range(number_of_iterations):
        print (f'\nIteration {iters+1} is running...\n')
        
        # Set the random seed for reproducibility
        seed = 1000+iters
        np.random.seed(seed)
        
        # CALL FUNCTION TO BUILD & COMPILE THE MODEL    
        if features==1:
            model = model_1D_cnn_1feature(num_classes,input_shape)
        elif features==2:
            model = model_2D_cnn_2features(num_classes,input_shape)
        elif features==3:
            model = model_2D_cnn_3features(num_classes,input_shape)
    
        train_data_pre, train_labels_pre, test_data, test_labels, class_weights = custom_split(data, labels, test_size, categories, seed)
        
        train_data, train_labels, test_labels = fix_training_set(train_data_pre,train_labels_pre,categories,test_labels,sampling,size_increased,balance,seed)
        
        # Z-SCORE NORMALIZATION (NORMALIZE ONLY THE VELOCITY AND Z-DEPTH SIGNAL)
        if len(train_data.shape)>2:
            for sh in range(1, train_data.shape[1]):
                train_data[:,sh,:] = (train_data[:,sh,:] - np.mean(train_data[:,sh,:]))/np.std(train_data[:,sh,:])
                test_data[:,sh,:] = (test_data[:,sh,:] - np.mean(test_data[:,sh,:]))/np.std(test_data[:,sh,:])
    
        norm_train_data = train_data
        norm_test_data = test_data
    
        # STORE TRAINING-TESTING SETS
        train_all_data_list.append(train_data)
        train_all_labels_list.append(train_labels)
        test_all_data_list.append(test_data)
        test_all_labels_list.append(test_labels)
    
        # PROCESS DATA & LABELS TO TRAIN AND TEST THE MODEL
        
        # Data need reshaping as the model gets input = [samples, timesteps, features]
        train_data_seq = np.expand_dims(norm_train_data,axis=-1)
        test_data_seq = np.expand_dims(norm_test_data,axis=-1)
            
        train_labels_seq = train_labels.reshape((len(norm_train_data),1))
        one_hot_lab_train = to_categorical(train_labels,num_classes)
        
        test_labels_seq = test_labels.reshape((len(norm_test_data),1))
        one_hot_lab_test = to_categorical(test_labels, num_classes)
        
        # SAVE MODEL
        model_filename = f'best_model_split_{iters+1}.hdf5'
        
        # model.save(path_dir+'model'+model_filename)
        es = EarlyStopping(monitor='val_accuracy', min_delta=0.005, mode='max', verbose=1, patience=200)
        mc = ModelCheckpoint(model_filename, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        tic = time.process_time() 
    
        class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
        
        # TRAIN THE MODEL
        model.fit(train_data_seq, one_hot_lab_train, batch_size=100,
                  epochs=epochs, validation_split=0.1,
                  class_weight=class_weights,callbacks=[es,mc])

        saved_model = load_model(model_filename)

        toc = time.process_time() 

        time_elapsed.append(toc-tic)

        # LOAD THE BEST MODEL SAVED

        # EVALUATE THE MODEL
        score = saved_model.evaluate(test_data_seq, one_hot_lab_test, batch_size=16)  # get loss & accuracy
        
        if score_old < score[1]:
            best_model_id = iters
            score_old = score[1]

        print(f'\nThe test loss and accuracy are {score[0]:0.2f} and {score[1]:0.2f}, respectively.\n')

        train_predictions = saved_model.predict_classes(train_data_seq)
        train_accuracy.append(accuracy_score(train_labels_seq, train_predictions))

        test_predictions = saved_model.predict_classes(test_data_seq)
        test_accuracy.append(accuracy_score(test_labels_seq, test_predictions))  # get accuracy

        train_all_predicitions_list.append(train_predictions)
        test_all_predicitions_list.append(test_predictions)
        
        #------------------------------------------------------------------------------------------------     
        
        # EVALUATE EACH CLASS IN TRAINING & TESTING SETS SEPERATELY
        train_conf_matr = confusion_matrix(train_labels, train_predictions, normalize='true')
        train_conf_matr_all.append(train_conf_matr)
        test_conf_matr = confusion_matrix(test_labels, test_predictions, normalize='true')
        test_conf_matr_all.append(test_conf_matr)

        for nclass in range(num_classes):

            if class_names[nclass] not in train_accuracies_all.keys():
                train_accuracies_all[class_names[nclass]] = [train_conf_matr[nclass,nclass]]
                test_accuracies_all[class_names[nclass]] = [test_conf_matr[nclass,nclass]]
            else:
                train_accuracies_all[class_names[nclass]].append(train_conf_matr[nclass,nclass])
                test_accuracies_all[class_names[nclass]].append(test_conf_matr[nclass,nclass])

    #------------------------------------------------------------------------------------------------ 

        # TRAINING/TESTING
        print("\nCONFUSION MATRICES\n")
        print("\nTraining set\n")
        print (confusion_matrix(train_labels, train_predictions, normalize='true'))
        print("\nTest set\n")
        print (confusion_matrix(test_labels, test_predictions, normalize='true'))

    #------------------------------------------------------------------------------------------------

    # SAVING THE RESULTS

    # EMPTY DICTIONARIES TO SAVE THE RESULTS
    mydict1 = {}
    # Training
    dict_train = {}
    dict_train['train_confusion_matrix'] = train_conf_matr_all
    dict_train['train_accur'] = train_accuracies_all

    # Testing
    dict_test = {}
    dict_test['test_confusion_matrix'] = test_conf_matr_all
    dict_test['test_accur'] = test_accuracies_all

    # All in one dictionary
    mydict1['training'] = dict_train
    mydict1['test'] = dict_test

    # SAVE PERFORMANCE
    pickle_filename_res='performance.pkl'

    with open(pickle_filename_res, 'wb') as file:
        pickle.dump(mydict1, file, protocol=pickle.HIGHEST_PROTOCOL)

    # SAVE TRAINING/TESTING SETS
    pickle_filename2='data.pkl'

    mydict2 = {} 
    mydict2['train_data']   = train_all_data_list
    mydict2['train_labels'] = train_all_labels_list
    mydict2['train_predictions'] = train_all_predicitions_list
    mydict2['test_data']    = test_all_data_list
    mydict2['test_labels']  = test_all_labels_list
    mydict2['test_predictions'] = test_all_predicitions_list
    with open(pickle_filename2, 'wb') as file:
        pickle.dump(mydict2, file, protocol=pickle.HIGHEST_PROTOCOL)

    #------------------------------------------------------------------------------------------------

    mean_time = np.mean(time_elapsed)

    print(f'\n The model run time approximattely {mean_time:.2f} seconds per iteration.')
    results = {}
    results['conf_matrix_mean'] = np.mean(test_conf_matr_all, axis=0)
    results['conf_matrix_std'] = np.std(test_conf_matr_all, axis=0)
    results['accuracy'] = test_accuracies_all
    results['class_names'] = class_names

    if plot:
        plots(results)       
    
    print(f'Best model is from iteration {best_model_id+1}')
    best_model = load_model(f'best_model_split_{best_model_id+1}.hdf5')

    return results, best_model


def data_preprocessing_laps_merging_classes(calcium_df_f, position, labels=None,
                                            step_laps=2, interp_timesteps=100,
                                            velocity=None, depth=None):
    """
    This function makes all the appropriate data preproccesing
    1. It loads all the data
    2. It removes all GOL experiment type from the data keeping only RandomForaging
    3. Removes all the nan data and those with unknown labels
    4. Breaks all calcium signals into laps (according to the animal position)
    5. Merges the laps per step_laps (parameter defined by the user)
    6. Removes all calcium signal laps with length<30 timesteps & max_position-min_position <0.95
    7. Interpolate all signal laps to be of equal length
    
    Args:
        calcium_df_f        : (list of np.arrays) Calcium signal DeltaF/F
        position            : (list of np.arrays) animal's position on the linear track  
        labels              : (list of str) labeling from immunochemistry 
                              'PV'    : Parvalbumin positive cells
                              'SOM'   : Somatostatin positive cells
                              'AAC'   : Axoaxonic cells
                              'BC'    : Basket cells
                              'CCK'   : CCK positive cells
                              'NPY'   : neuroglia form cells
                              'AAC/BC': Axoaxonic or basket cells
                              'BISTR' : Bistratified cells
                              'U'     : Unspecified
        step_laps           : (int) laps to be merged
        interp_timesteps    : (int) time-steps after interpolation
        velocity            : (list of np.arrays) velocity of the animal, Default None
        depth               : (list of floats) depth of a cell, Default None
            
    Returns:data_laps, labels_final, features
        data_laps           : (np.array) data pooled together. When DF/F is
                              given, a 2D array (number of samples x interp_timesteps).
                              When more features are added a 3D array
                              is returned (number of samples x features x interp_timesteps)
        labels_final        : (np.array) labels as str (see above)
        features            : (int) number of features
    """

    import numpy as np
    from scipy import interpolate
    
    cal_data_laps = []
    vel_data_laps = []
    depth_data_laps = []
    pos_data_laps = []
    label_laps = []

    N = len(calcium_df_f)         
#------------------------------------------------------------------------------------------------    

    for datum in range(N):
        cal_data = calcium_df_f[datum]
        pos_data = position[datum]
        
        if labels:
            label_data = labels[datum]

        if labels:
            if label_data=='U':
                continue

        if np.isnan(cal_data).any():
            continue

        if np.isnan(pos_data).any():
            continue

        if velocity:
            vel_data = velocity[datum]

            if np.isnan(vel_data).any():
                continue

        if depth:
            depth_data = depth[datum]

            if np.isnan(depth_data).any():
                continue

        # Find the start/end of each lap and store them in a list
        laps = np.where(np.diff(pos_data)<-0.5)
        laps_last_elem_app = np.append(laps, len(pos_data))

        # First two laps are removed
        removed_laps = 2
        total_laps = len(laps_last_elem_app)-removed_laps
        # Merge laps per step_laps, construction of pointer arrays/lists
        mod = total_laps % step_laps
        laps_first_elem_app = np.array([0] + [jj+1 for jj in laps_last_elem_app[:-1]])
        laps_idxs = [(iii, jjj) for iii,jjj in zip(laps_first_elem_app, laps_last_elem_app)]  
        
        # REMOVE FIRST 2 LAPS
        laps_idxs_remove = laps_idxs[removed_laps:]

        # Remove if necessary laps from the beginning depending on the step size                
        laps_idxs_final = laps_idxs_remove[mod:]

        idxs_merge = [iiii for iiii in range(0,len(laps_idxs_final)-1, step_laps)]

        for l in idxs_merge:

            t_lap_start = laps_idxs_final[l][0]
            t_lap_end   = laps_idxs_final[l+step_laps-1][1]

            cal_data_per_lap = cal_data[t_lap_start:t_lap_end+1]
            cal_data_laps.append(cal_data_per_lap)
 
            pos_data_per_lap = pos_data[t_lap_start:t_lap_end+1]
            pos_data_laps.append(pos_data_per_lap)

            if velocity:
                vel_data_per_lap = vel_data[t_lap_start:t_lap_end+1]
                vel_data_laps.append(vel_data_per_lap)
            if depth:
                depth_data_per_lap = depth_data*np.ones(len(pos_data_per_lap))
                depth_data_laps.append(depth_data_per_lap)
          
            if labels:
                label_laps.append(label_data)

    # REMOVE ALL CALCIUM SIGNAL LAPS WITH LENGTH<30 TIMESTEPS & MAX_POSITION-MIN_POSITION<0.95
    diffs = [max(i)-min(i) for i in pos_data_laps]  
    diffs = np.asarray(diffs)

    full_lap = np.where(diffs>=0.95)[0]

    cal_data_full_lap = []
    vel_data_full_lap = []
    depth_data_full_lap = []
    labels_full_lap = []

    for i in range(len(full_lap)):
        if len(cal_data_laps[full_lap[i]])>=30:
            cal_data_full_lap.append(cal_data_laps[full_lap[i]])

            if velocity:
                vel_data_full_lap.append(vel_data_laps[full_lap[i]])
            if depth:
                depth_data_full_lap.append(depth_data_laps[full_lap[i]])

            if labels:
                labels_full_lap.append(label_laps[full_lap[i]])
    if labels:
        labels_full_lap = np.asarray(labels_full_lap)  

#------------------------------------------------------------------------------------------------        

    # INTERPOLATE SIGNALS                                           
    interp_cal_data = []
    interp_vel_data = []
    interp_dep_data = []

    for q in range(len(cal_data_full_lap)):

        time_steps = len(cal_data_full_lap[q])
        xactual = np.linspace(0,time_steps,time_steps)
        xvals = np.linspace(0, time_steps, interp_timesteps)

        interpf_cal_data = interpolate.interp1d(xactual,cal_data_full_lap[q], kind='cubic')
        interp_cal_data.append(interpf_cal_data(xvals))

        if velocity:
            interpf_vel_data = interpolate.interp1d(xactual,vel_data_full_lap[q], kind='cubic')
            interp_vel_data.append(interpf_vel_data(xvals))

        if depth:
            interpf_dep_data = interpolate.interp1d(xactual,depth_data_full_lap[q], kind='cubic')
            interp_dep_data.append(interpf_dep_data(xvals))

#------------------------------------------------------------------------------------------------
    cal_laps_final = np.asarray(interp_cal_data)
    vel_laps_final = np.asarray(interp_vel_data)
    dep_laps_final = np.asarray(interp_dep_data)

    if labels:
        labels_final = labels_full_lap
    else:
        labels_final = []

    if velocity and depth:
        data_laps = np.concatenate((np.expand_dims(cal_laps_final,axis=1),
                                    np.expand_dims(vel_laps_final,axis=1),
                                    np.expand_dims(dep_laps_final,axis=1)),
                                   axis=1)             
        features = 3

    elif velocity and not depth:
        data_laps = np.concatenate((np.expand_dims(cal_laps_final,axis=1),
                                    np.expand_dims(vel_laps_final,axis=1)),
                                   axis=1)
        features = 2

    elif depth and not velocity:
        data_laps = np.concatenate((np.expand_dims(cal_laps_final,axis=1),
                                    np.expand_dims(dep_laps_final,axis=1)),
                                   axis=1)
        features = 2

    else:
        data_laps = cal_laps_final
        features = 1

    return data_laps, labels_final, features


def model_1D_cnn_1feature(num_classes, input_shape):
    """
    This function builds and compiles the 1-Dimensional CNN model
    
    Args:
        num_classes : (int) number of classes
        input_shape : (np.array) the shape of the input vector as per Keras
        
    Returns:
        model       : (keras.engine.sequential.Sequential) the model to be trained
    
    """
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    from keras import regularizers, optimizers
    from keras.layers.normalization import BatchNormalization


    # DEFINE CONSTRUCTOR
    model = Sequential()

    # BUILD THE MODEL
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu',padding='valid'))

    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3,strides=2))

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3,strides=2))

    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3,strides=2))

    model.add(Flatten())
    model.add(Dropout(0.5)) 

    model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l2(0.0),
            activity_regularizer=regularizers.l2(0.0)))

    # COMPILE THE MODEL
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)    

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def model_2D_cnn_2features(num_classes, input_shape):
    """
    This function builds and compiles the 2-Dimensional CNN model
    
    Args:
        num_classes : (int) number of classes
        input_shape : (np.array) the shape of the input vector as per Keras
        
    Returns:
        model       : (keras.engine.sequential.Sequential) the model to be trained
    
    """
    # This function builds and compiles the 2-Dimensional CNN model when using 2 features (calcium-velocity)
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
    from keras import optimizers
    from keras.layers.normalization import BatchNormalization

    # DEFINE CONSTRUCTOR
    model = Sequential()

    # BUILD THE MODEL
    model.add(Conv2D(filters=256, kernel_size=(2,3), activation='relu', padding='valid', input_shape=input_shape))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size=(1,3), activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1,3), strides=(1,2),padding='same'))
    
    model.add(Conv2D(filters=64, kernel_size=(1,3), activation='relu',padding='same'))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1,3), strides=(1,2),padding='same'))

    model.add(Conv2D(filters=32, kernel_size=(1,3), activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1,3),strides=(1,2),padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.5))  

    model.add(Dense(num_classes, activation='softmax'))

    # COMPILE THE MODEL
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def model_2D_cnn_3features(num_classes, input_shape):
    """
    This function builds and compiles the 2-Dimensional CNN model
    
    Args:
        num_classes : (int) number of classes
        input_shape : (np.array) the shape of the input vector as per Keras
        
    Returns:
        model       : (keras.engine.sequential.Sequential) the model to be trained
    
    """
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
    from keras import optimizers
    from keras.layers.normalization import BatchNormalization

    # DEFINE CONSTRUCTOR
    model = Sequential()

    # BUILD THE MODEL
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='valid', input_shape=input_shape))
    model.add(BatchNormalization()) 

    model.add(Conv2D(filters=128, kernel_size=(1,3), activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1,3), strides=(1,2),padding='same'))

    model.add(Conv2D(filters=64, kernel_size=(1,3), activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1,3), strides=(1,2),padding='same'))

    model.add(Conv2D(filters=32, kernel_size=(1,3), activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1,3),strides=(1,2),padding='same'))

    model.add(Flatten())
    
    model.add(Dropout(0.5)) 

    model.add(Dense(num_classes, activation='softmax'))

    # COMPILE THE MODEL
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model


def custom_split(data, labels, test_size, indexing, seed):
    """
    This function performs a custom split into training and test data
    
    Args:
        data         : (np.array) A 3D or 2D matrix with all data
        labels       : (np.array) A 1D vector with the labels
        indexing     : (list of lists of str) categorization of the cells
        seed         : (int) random seed generator
        
    Returns:
        train_data    : (np.array) A 3D or 2D matrix with the training data
        train_labels  : (np.array) A 1D vector with the training labels
        test_data     : (np.array) A 3D or 2D matrix with the test data
        test_labels   : (np.array) A 1D vector with the test labels
        class_weights : (np.array) A 1D vector (num_classes, ) with the
                        weights of each class to account for the unbalanced
                        class sizes
    
    """    
    import numpy as np
    from sklearn.utils import class_weight

    if data.shape[0]!=labels.shape[0]:
        raise ValueError('Sizes of data and labels do not match.')

    np.random.seed(seed)

    num_classes = int(len(np.unique(labels)))

    test_data = []
    test_labels = []
    rest_data = []
    rest_labels = []

    real_index = []
    for i in indexing:
        real_index+=i

    real_test = []
    for i in test_size:
        real_test+=i

    for i in range(num_classes):

        nclass = real_index[i]
        label_idxs = np.where(labels==nclass)[0]
        test_idxs = np.random.choice(label_idxs, real_test[i], replace=False)
        rest_idxs = np.array([jj for jj in list(label_idxs) if jj not in list(test_idxs)])

        if len(test_data)==0 and len(test_labels)==0:
            rest_data = data[rest_idxs,:]
            rest_labels = labels[rest_idxs]
            test_data = data[test_idxs,:]
            test_labels = labels[test_idxs]

        else:
            rest_data = np.concatenate((rest_data, data[rest_idxs,:]), axis=0)
            rest_labels = np.concatenate((rest_labels,labels[rest_idxs]))                
            test_data = np.concatenate((test_data, data[test_idxs,:]), axis=0)
            test_labels = np.concatenate((test_labels,labels[test_idxs]))

    train_data = rest_data
    train_labels = rest_labels
    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)

    return (train_data, train_labels, test_data, test_labels, class_weights)


def take_labels(data_all, labels_all, indexing):
    """
    Extract the labels from a given data set

    Args:
        data_all         : (np.array) all data points
        labels_all       : (np.array) all labels
        indexing         : (list of lists of str) categories to be used/merged

    Returns:
        data             : (np.array) data based on the categories
        labels           : (np.array) corresponding labels

    """

    import numpy as np

    data = []
    for index in indexing:
        if len(index)!=1:
            for idx in index:
                indices = np.where(labels_all==idx)[0]

                if len(data)==0:
                    data = data_all[indices]
                    labels= labels_all[indices]
                else:
                    data = np.concatenate((data, data_all[indices]),axis=0)
                    labels = np.concatenate((labels,labels_all[indices]))
        else:
            idx = index[0]
            indices = np.where(labels_all==idx)[0]
            if len(data)==0:
                data = data_all[indices]
                labels = labels_all[indices]
            else:
                data = np.concatenate((data, data_all[indices]),axis=0)
                labels = np.concatenate((labels, labels_all[indices]))

    return (data, labels)


def fix_training_set(train_data_pre, train_labels_pre, indexing, test_labels,
                     sampling, size_increased, balance, seed):
    """
    Function to modify the training set samples and to tranform the str labels into int
    
    Args:
        train_data_pre   : (np.array) training data
        train_labels_pre : (np.array) training labels, str
        indexing         : (list of lists of srt) categorization
        test_labels      : (np.array) test labels, str
        sampling         : (str) specify the type of sampling in the training set
                            imbalanced    : all data of each category (no downsample)
                            semi-balanced : variable sizes (downsampling)
                            min_categ     : downsample to match the minimum sized category
        size_increased   : (list of int) size of each neuronal category in
                           in the training dataset
        balance          : (str) In case of merged categories either use equal
                           size of each subcategory or keep the same precentage
                           as in the original dataset.Valid options: equal, stratified
        seed             : (int) random seed

    Returns:
        train_data       : (np.array) updated training data
        train_labels     : (np.array) updated training labels, integers
        test_labels      : (np.array) updated test labels, integers

    """

    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    from functions import categories_balance

    sizes = categories_balance(train_data_pre,train_labels_pre,indexing,sampling,size_increased)
    if sampling!='imbalanced':
        train_data = []
        for j in range(len(indexing)):
            index = indexing[j]
            if len(index)!=1:
                train_idxs = []
                for idx in index:
                    if len(train_idxs)==0:
                        train_idxs = np.where(train_labels_pre==idx)[0]
                    else:
                        train_idxs = np.concatenate((train_idxs, np.where(train_labels_pre==idx)[0]))

                train_data_post = train_data_pre[train_idxs]
                train_labels_post = train_labels_pre[train_idxs]
                if balance=='stratified':
                    train_data_post2, test_data_post2,train_labels_post2, test_labels_post = train_test_split(train_data_post, train_labels_post, train_size=sizes[j], random_state = seed, stratify=train_labels_post)
                elif balance=='equal':
                    size_i = int(sizes[j]/len(index))
                    train_data_post2   = []
                    train_labels_post2 = []
                    for i in range(len(index)):
                        if len(np.where(train_labels_post==index[i])[0])<size_i:
                            raise ValueError(f'{idx} has not enough training examples. Reduce the corresponding value in size_increased.')
                        train_post_idx_i = np.random.choice(np.where(train_labels_post==index[i])[0], size_i, replace=False)
                        if len(train_data_post2)==0:
                            train_data_post2 = train_data_post[train_post_idx_i]
                            train_labels_post2 = train_labels_post[train_post_idx_i]
                        else:
                            train_data_post2 = np.concatenate((train_data_post2,train_data_post[train_post_idx_i]),axis=0)
                            train_labels_post2 = np.concatenate((train_labels_post2, train_labels_post[train_post_idx_i]),axis=0)
                else:
                    raise ValueError(r"Not a valid balance argument. Use either 'stratified' or 'equal'.")
                if len(train_data)==0:
                    train_data = train_data_post2.copy()
                    train_labels = train_labels_post2.copy()
                else:
                    train_data = np.concatenate((train_data,train_data_post2), axis=0)
                    train_labels = np.concatenate((train_labels,train_labels_post2))
            else:
                idx = index[0]
                train_idxs_all = np.where(train_labels_pre==idx)[0]
                if len(train_idxs_all)<sizes[j]:
                    raise ValueError(f'{idx} has not enough training examples. Reduce the corresponding value in size_increased.')
                train_idxs = np.random.choice(train_idxs_all, sizes[j], replace=False)
                train_data_post3 = train_data_pre[train_idxs]
                train_labels_post3 = train_labels_pre[train_idxs]
                if len(train_data)==0:
                    train_data = train_data_post3.copy()
                    train_labels = train_labels_post3.copy()
                else:
                    train_data = np.concatenate((train_data,train_data_post3), axis=0)
                    train_labels = np.concatenate((train_labels,train_labels_post3))
    else:
        train_data = train_data_pre
        train_labels = train_labels_pre
    
    for i in range(len(indexing)):
        index = indexing[i]
        if len(index)!=1:
            for idx in index:
                train_labels[np.where(train_labels==idx)[0]]=i
                test_labels[np.where(test_labels==idx)[0]]=i
        else:
            idx=index[0]
            train_labels[np.where(train_labels==idx)[0]]=i
            test_labels[np.where(test_labels==idx)[0]]=i

    train_labels = train_labels.astype(float)
    test_labels = test_labels.astype(float)
    train_data, train_labels = shuffle(train_data, train_labels, random_state=seed)

    return (train_data, train_labels, test_labels)


def categories_balance(train_data_pre, train_labels_pre, indexing, sampling,
                       size_increased):
    """
    Function to specify if the training set is imbalanced, balanced or semi balanced
    Args:
        train_data_pre   : (np.array) training data
        train_labels_pre : (np.array) training labels
        indexing         : (list of lists of srt) categorization
        sampling         : (str) specify the type of sampling in the training set
                            imbalanced    : all data of each category (no downsampling)
                            semi-balanced : variable sizes (downsampling)
                            min_categ     : downsample to match the minimum sized category
        size_increased   : (list of int) size of each neuronal category in
                           in the training dataset

    Returns:
        sizes            : (list) number of samples per class, int

    """
    import numpy as np

    num_classes = len(indexing)
    numbers = []
    for index in indexing:
        if len(index)!=1:
            sub = []
            for idx in index:
                sub.append(train_data_pre[train_labels_pre==idx].shape[0])
            numbers.append(sub)
        else:
            idx=index[0]
            numbers.append([train_data_pre[train_labels_pre==index[0]].shape[0]])

    total = [sum(i) for i in numbers]    

    if sampling!='imbalanced':
    
        if sampling=='min_categ':
            sizes = len(total)*[np.min(total)]
        elif sampling=='semi_balanced':
            sizes = []
            for i in range(num_classes):
                minimum = int(np.min(total))
                sizes.append(minimum + size_increased[i])
        else:
            raise ValueError(r"Not valid sampling approach. Use 'imbalanced', 'semi_balanced', or 'min_categ'.")        
    elif sampling=='imbalanced':
        sizes = numbers

    return (sizes)


def plots(results):
    """
    Parameters:
        results : (dict) dictionary with the output of the classifier

    Returns:
        None.
        
    """

    import numpy as np
    import seaborn as sn
    import matplotlib.pyplot as plt
    from sympy import Symbol

    data = []
    names = []
    for name in results['class_names']:
        data+=results['accuracy'][name]
        names+=[name]*len(results['accuracy'][name])

    names_new = ['PVBC\nAAC', 'SomC\nBiC', 'CCKC', 'IvC\nNGFC']
    plt.figure(figsize=(10,7))
    sn.boxplot(x=names, y=data)
    plt.axhline(y=1/(results['conf_matrix_std'].shape[0]),xmin=0,xmax=4, color='red', linestyle='--',label='chance level')
    sn.set(font_scale=1.4)
    # plt.xlabel('category')
    plt.ylabel('Classification accuracy')
    sn.despine(offset=10, trim=True)
    plt.legend()
    plt.xticks(range(4), names_new)
    
    anots = []# np.chararray((anots.shape[0], anots.shape[1]), unicode=False)
    pm = Symbol(u'±')

    for m in range(results['conf_matrix_std'].shape[0]):
        for n in range(results['conf_matrix_std'].shape[1]):
            anots.append( f"{results['conf_matrix_mean'][m,n]*100:.1f}{pm}{results['conf_matrix_std'][m,n]*100:.1f}%")
    anots = np.asarray(anots).reshape(results['conf_matrix_std'].shape[0],results['conf_matrix_std'].shape[0])
    plt.figure(figsize=(14,9))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(results['conf_matrix_mean'], annot=anots, 
               annot_kws={"size": 14},
               cmap='jet', square=True, fmt='s', vmin=0, vmax=0.7,
               yticklabels=names_new, xticklabels=names_new,
               cbar_kws={'label': 'accuracy'})
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    # plt.xlabel('predicted label')
    # plt.ylabel('true label')
    

def predictions(df_f, new_df_f, position, new_position, labels, categories, test_size,
                sampling='imbalanced', balance='stratified',
                epochs=200, number_of_iterations=10,
                size_increased=None, velocity=None, depth=None, plot=True,
                new_velocity=None, new_depth=None, new_labels=None):
    """
    

    Args:
        calcium_df_f          : (list of np.arrays) Calcium signal DeltaF/F
        new_df_f              : (list of np.arrays) Calcium signal DeltaF/F for prediction
        position              : (list of np.arrays) animal's position on the linear track
        new_position          : (list of np.arrays) animal's position for prediction
        labels                : (list of str) labeling from immunochemistry 
                                'SOM'   : Somatostatin positive cells
                                'AAC'   : Axoaxonic cells
                                'BC'    : Basket cells
                                'CCK'   : CCK positive cells
                                'NPY'   : neuroglia form cells
                                'BISTR' : Bistratified cells
                                'U'     : Unspecified
        categories            : (list of lists of str) categorization of cells.
                                In case of a list with more than one type, these
                                categories are merged in one super-category
        test_size             : (list of lists of int) The test size per category
                                in order for every iteration to be comparable with
                                the others.
        sampling              : (str) specify the type of sampling in the training set
                                imbalanced    : all data of each category (no subsampling)
                                semi-balanced : variable sizes (downsampling)
                                min_categ     : downsample to match the minimum sized category
        balance               : (str) In case of merged categories either use equal
                                size of each subcategory or keep the same precentage
                                as in the original dataset.Valid options: equal, stratified
        epochs                : (int) number of maximum epochs for classifier. Default is 100.
        number_of_iterations  : (int) number of random train-test splits. Default is 10.
        size_increased        : (list of int) size of each neuronal type in
                                in the training dataset above the mionimum category
                                Optional, Default is None
        velocity              : (list of floats) animal's velocity, Default None
        depth                 : (list of floats) depth of a cell, Default None
        plot                  : (boolean) True (default) for plotting, False othwerwise.
        new_velocity          : (list of floats) new animal's velocity, Default None
        new_depth             : (list of floats) new depth of a cell, Default None
        new_labels            : (list of str) labeling from immunochemistry, if any (for testing the algorithm)
    Returns:
        test_predictions_str  : (np.array of str) Predicted labels, following the same notation
    """
    
    import numpy as np
    from functions import CNN_classifier, take_labels
    from sklearn.metrics import confusion_matrix

    r, model =  CNN_classifier(calcium_df_f=df_f, position=position, labels=labels,
                               categories=categories, test_size=test_size,
                               sampling=sampling, balance=balance, epochs=epochs,
                               number_of_iterations=number_of_iterations, size_increased=size_increased,
                               velocity=velocity, depth=depth, plot=False)

    # GET ALL DATA
    return_list = data_preprocessing_laps_merging_classes(calcium_df_f=new_df_f, position=new_position, labels=new_labels,
                                                step_laps=2, interp_timesteps=100,
                                                velocity=new_velocity, depth=new_depth)
    
    new_data = return_list[0]
    if new_labels:
        labels_ = return_list[1]
    else:
        labels_ = []
    features = return_list[2]

    if new_labels:
        new_data_, labels_ = take_labels(new_data, labels_, categories)
        
        # Merge categories
        for i in range(len(categories)):
            index = categories[i]
            if len(index)!=1:
                for idx in index:
                    labels_[np.where(labels_==idx)[0]]=i
            else:
                idx=index[0]
                labels_[np.where(labels_==idx)[0]]=i

    if features>1:
        if model.input_shape[1] != features:
            raise ValueError('Wrong test data format and/or trained model!')

        
    # Z-SCORE NORMALIZATION (NORMALIZE ONLY THE VELOCITY AND Z-DEPTH SIGNAL)
    if len(new_data_.shape)>2:
        for sh in range(1, new_data_.shape[1]):
            new_data_[:,sh,:] = (new_data_[:,sh,:] - np.mean(new_data_[:,sh,:]))/np.std(new_data_[:,sh,:])

    a = list(range(len(labels_)))
    np.random.shuffle(a)
    a = np.array(a)
    norm_new_data = new_data_[a]
    labels_ = labels_[a]

    # PROCESS DATA TO TRAIN AND TEST THE MODEL
    # Data need reshaping as the model gets input = [samples, timesteps, features]
    new_data_seq = np.expand_dims(norm_new_data,axis=-1)

    test_predictions = model.predict_classes(new_data_seq)
    
    test_conf_matr = confusion_matrix(labels_.astype(int), test_predictions, normalize='true')
    
    M = np.max(test_predictions)
    test_predictions_str = np.zeros_like(test_predictions).astype(str)
    labels_str = np.zeros_like(labels_).astype(str)
    for nn in range(M+1):
        test_predictions_str[test_predictions==nn] = r['class_names'][nn]
        if new_labels:
            labels_str[labels_.astype(int)==nn] = r['class_names'][nn]

    return test_predictions_str, test_conf_matr
    
