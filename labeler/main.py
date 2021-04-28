from create_dataset import create_dataset
from train_model import train_model
from predict import predict

import numpy as np
from tensorflow import keras

class main:
    """ Create train dataset, using labeled videos, and store it in dataset/ """
    # Comment next line if datasets have already been generated
    create_dataset.create_dataset_using_labeled_videos()

    """ Train ANN model using previously generated train dataset """    
    # Retrieve saved datasets
    x = np.load('dataset/train_set.npy')
    y = np.load('dataset/yaws_pitches.npy')
    # Train model
    ann_trained_model = train_model.train_ann(x, y)
    # Save model
    ann_trained_model.save('models/trained_model')
    """ Predict using trained model """
    predict.ann_predict(keras.models.load_model('models/trained_model'), 5)
