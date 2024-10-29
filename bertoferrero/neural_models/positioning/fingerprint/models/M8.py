# Copyright 2024 Alberto Ferrero López
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import tensorflow as tf
from bertoferrero.neural_models.positioning.fingerprint.trainingcommon import load_data
from .ModelsBaseClass import ModelsBaseClass


class M8(ModelsBaseClass):
    @staticmethod
    def load_traning_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, train_scaler_file=True, include_pos_z=False, scale_y=False)

    @staticmethod
    def load_testing_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, train_scaler_file=False, include_pos_z=False, scale_y=False)

    def build_model(self, empty_values: bool = False):
        pass

    def build_model_autokeras(self, designing: bool, overwrite: bool, tuner: str, random_seed: int, autokeras_project_name: str, auokeras_folder: str, max_trials: int = 100):
        pass

    def build_custom_model(self, dimension1: dict, dimension2: dict, loss_weight_d1: float, loss_weight_d2: float, inputdropout: float, learning_rate: float, loss: str, activation_d1: str = 'softmax', activation_d2: str = 'softmax'):
        input_rssi = tf.keras.Input(shape=(self.inputlength,), name='input_rssi')
        hiddenLayers = tf.keras.layers.Dropout(
            inputdropout, name='input_dropout')(input_rssi)

        # Añadimos las capas de la dimension 1
        dimension_number = 1
        layer_number = 1
        for layer in dimension1:
            # Extraemos configuración
            dimension_units = layer['units']
            dimension_dropout = layer['dropout']

            # Añadimos capa
            hiddenLayers = tf.keras.layers.Dense(
                dimension_units, activation='relu', name='dense_'+str(dimension_number)+'.'+str(layer_number))(hiddenLayers)
            hiddenLayers = tf.keras.layers.Dropout(
                dimension_dropout)(hiddenLayers)

            # Incrementamos contador
            layer_number += 1

        # Creamos el output de la dimension 1
        output_d1 = tf.keras.layers.Dense(
            units=self.outputlength[0], activation=activation_d1, name='output_d1')(hiddenLayers)

        # Añadimos las capas de la dimension 2
        dimension_number = 2
        layer_number = 1
        for layer in dimension2:
            # Extraemos configuración
            dimension_units = layer['units']
            dimension_dropout = layer['dropout']

            # Añadimos capa
            if(layer_number == 1):
                hiddenLayers = tf.keras.layers.Dense(
                    dimension_units, activation='relu', name='dense_'+str(dimension_number)+'.'+str(layer_number))(output_d1)
            else:
                hiddenLayers = tf.keras.layers.Dense(
                    dimension_units, activation='relu', name='dense_'+str(dimension_number)+'.'+str(layer_number))(hiddenLayers)
            hiddenLayers = tf.keras.layers.Dropout(
                dimension_dropout)(hiddenLayers)

            # Incrementamos contador
            layer_number += 1

        # Creamos el output de la dimension 2
        output_d2 = tf.keras.layers.Dense(
            self.outputlength[1], activation=activation_d2, name='output_d2')(hiddenLayers)

        # Creamos el modelo
        model = tf.keras.Model(
            inputs=[input_rssi],
            outputs=[output_d1, output_d2]
        )
        # Optimizador
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss=loss, optimizer=optimizer, metrics=[
                      'accuracy'], loss_weights=[loss_weight_d1, loss_weight_d2])

        return model
