import cv2

import tensorflow as tf

class params:

    """
    Create dataset params
    """

    n_of_features_per_row = 1

    # Paths
    root_dir_labeled = "../labeled/"
    root_dir_dataset = "dataset/"

    
    """
    CV params
    """

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=0, qualityLevel=0.01, minDistance=15, blockSize=5)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    """
    Model params
    """

    n_of_epochs = 10000
    n_of_neurons = 1
    n_of_output = 1
    
    size_of_batch = 32

    activation_function_input = 'relu'
    activation_function_output = 'linear'


    optimizer_type = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False, name="SGD")
    loss_type = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
    metrics_types = [tf.keras.metrics.MeanSquaredError()]

    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="mean_squared_error", min_delta=0, patience=1000, verbose=1, mode="min", baseline=None, restore_best_weights=True)
    callbacks = [earlyStopping]


