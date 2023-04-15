from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


import os
import math

from sklearn.model_selection import train_test_split

def masked_mean_sequare_error_sw(y_true, y_pred):
    delta_y = y_pred - y_true
    delta_y = tf.where(tf.math.is_nan(delta_y), tf.zeros_like(delta_y), delta_y)
    
    squared_difference = tf.square(delta_y)
    
    return tf.reduce_mean(squared_difference)


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        top_model = False,
        regression_target = None,
        batch_size: int = 32,
        n_steps_per_epoch: int = 100,
        shuffle: bool = True
    ):

        self.data = data
        self.batch_size = batch_size
        self.n_steps_per_epoch = n_steps_per_epoch
        self.shuffle = shuffle
        self.top_model = top_model
        
        if self.shuffle:
            self.step_batch_idx = np.random.choice(len(self.data), size=(self.n_steps_per_epoch, self.batch_size))
        else:
            self.step_batch_idx = np.arange(len(self.data)).reshape((1, -1))
        
        if self.top_model:
            self.regression_target = regression_target
            
            
    def __len__(self):
        return self.n_steps_per_epoch
        
    def __getitem__(self, i_batch):
        X, y = self.__data_generation(i_batch)
        return X['input_data'], y
        
    def on_epoch_end(self):
        if self.shuffle:

            self.step_batch_idx = np.random.choice(len(self.data), size=(self.n_steps_per_epoch, self.batch_size))
        else:
            self.step_batch_idx = np.arange(len(self.data)).reshape((1, -1))
        
    def __data_generation(self, i_batch):
        batch_idx = self.step_batch_idx[i_batch]
        X = np.zeros((self.batch_size, self.data.shape[1]), dtype=np.float32)
        regression_values = []
        y_dict = {}
        for i_data, idx in enumerate(batch_idx):
            if self.top_model:
                regression_values.append(self.regression_target[idx])
            
            X[i_data, :] = self.data[idx]
            
        if self.top_model:
            y_dict['Regression'] = np.asarray(regression_values)
        y_dict['Autoencoder'] = X.copy()
        
        return (dict(input_data=X), y_dict)

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape = (batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VariationalAutoencoderWithTop(tf.keras.Model):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 top_model,
                 type_of_vae,
                 weights_dict,
                 **kwargs):
        super(VariationalAutoencoderWithTop,self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.weights_dict = weights_dict
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name = "reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name = "kl_loss")
        self.regression_loss_tracker = keras.metrics.Mean(name = "regression_loss")
        self.top_model = top_model
        self.type_of_vae = type_of_vae
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.regression_loss_tracker
        ]
    
    @staticmethod
    def compute_mmd(x, y):
        """
        Maximum Mean Discrepancy loss
        reference: https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
        """

        def compute_kernel(x, y):
            x_size = tf.shape(x)[0]
            y_size = tf.shape(y)[0]
            dim = tf.shape(x)[1]
            tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
            tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
            return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

        x_kernel = compute_kernel(x, x)
        y_kernel = compute_kernel(y, y)
        xy_kernel = compute_kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            X, y = data
            encoder_output = self.encoder(X)
            z = encoder_output['z_kld']
            if self.top_model:
                z_reg_model = encoder_output['z_reg_model']
            z_kld = encoder_output['z_kld']
            recon = self.decoder(z)
            
            reconstruction_loss = masked_mean_sequare_error_sw(y['Autoencoder'], recon)
            if self.top_model:
                y_true = tf.reshape(y['Regression'], (len(y['Regression']),1))
                y_pred = tf.reshape(z_reg_model, (len(y['Regression']),1))
                
                regression_loss = masked_mean_sequare_error_sw(y_true, y_pred)
            else:
                regression_loss = 0
            
            if self.type_of_vae =="ELBO":
                z_mean = encoder_output['z_mean']
                z_log_var = encoder_output['z_log_var']
                kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1))
                

            else:
                
                gaussian_sample = tf.random.normal(shape = (tf.shape(z_kld)[0], tf.shape(z_kld)[1]), mean = 0, stddev = 1)
                kl_loss = self.compute_mmd(gaussian_sample, z_kld)

            total_loss = self.weights_dict['kl_loss']*kl_loss + self.weights_dict['recon_loss']*reconstruction_loss + self.weights_dict['regression_loss']*regression_loss

            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.regression_loss_tracker.update_state(regression_loss)
        return {
            "total_loss" : self.total_loss_tracker.result(),
            "reconstruction_loss":self.reconstruction_loss_tracker.result(),
            "kl_loss_tracker":self.kl_loss_tracker.result(),
            "regression_loss_tracker":self.regression_loss_tracker.result()
        }
    
class WindowGenerator():
    def __init__(self, 
                 input_width, 
                 label_width,
                 shift,
                 train_df, 
                 val_df, 
                 test_df,
               label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data, shuffled, batch_size):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
              data=data,
              targets=None,
              sequence_length=self.total_window_size,
              sequence_stride=1,
              shuffle=shuffled,
              batch_size=batch_size)

        ds = ds.map(self.split_window)

        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df, True, 32)

    @property
    def val(self):
        return self.make_dataset(self.val_df, True, 32)

    @property
    def test(self):
        return self.make_dataset(self.test_df, True, 32)
    
    
