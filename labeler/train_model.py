import tensorflow as tf
from keras.callbacks import EarlyStopping

from params import params

tf.__version__

class train_model:

    
    n_of_epochs = params.n_of_epochs
    n_of_neurons = params.n_of_neurons
    n_of_output = params.n_of_output
    
    size_of_batch = params.size_of_batch

    activation_function_input = params.activation_function_input
    activation_function_output = params.activation_function_output


    optimizer_type = params.optimizer_type
    loss_type = params.loss_type
    metrics_types = params.metrics_types

    earlyStopping = params.earlyStopping
    callbacks = params.callbacks


    def train_ann(train_set_complete, desired_output):
        ann = tf.keras.models.Sequential()
        ann.add(tf.keras.layers.Dense(units=train_model.n_of_neurons, activation=train_model.activation_function_input, input_shape=(train_set_complete.shape)))
        ann.add(tf.keras.layers.Dense(units=train_model.n_of_output, activation=train_model.activation_function_output))
        ann.compile(optimizer=train_model.optimizer_type, loss=train_model.loss_type, metrics=train_model.metrics_types)
        
        ann.fit(train_set_complete, desired_output, batch_size=train_model.size_of_batch, epochs=train_model.n_of_epochs, callbacks=train_model.callbacks)

        return ann
