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
import random
import autokeras as ak
from bertoferrero.neural_models.positioning.fingerprint.trainingcommon import load_data
from .ModelsBaseClass import ModelsBaseClass

class M4(ModelsBaseClass): 
    @staticmethod
    def load_traning_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, train_scaler_file=True, include_pos_z=False,
                        scale_y=True, not_valid_sensor_value=100, return_valid_sensors_map=True)

    @staticmethod
    def load_testing_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, include_pos_z=False, scale_y=True, not_valid_sensor_value=100, return_valid_sensors_map=True)

    def build_model(self, random_seed:int, empty_values: bool = False, base_model_path: str = None):
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        #Cargamos el modelo partiendo de una base o integramente desde la api
        if base_model_path is not None:
            return self._build_model_from_base(base_model_path=base_model_path, loss='mse', metrics=['mse', 'accuracy'])
        
        if not empty_values:
            raise NotImplementedError("Not empty values model building is not implemented yet.")
        
        input_1 = tf.keras.layers.Input(shape=(self.inputlength,), name='input_1')
        dense = tf.keras.layers.Dense(64, activation='linear', name='dense')(input_1)
        re_lu = tf.keras.layers.ReLU(name='re_lu')(dense)

        input_2 = tf.keras.layers.Input(shape=(self.inputlength,), name='input_2')
        dense_1 = tf.keras.layers.Dense(16, activation='linear', name='dense_1')(input_2)
        re_lu_1 = tf.keras.layers.ReLU(name='re_lu_1')(dense_1)

        dense_2 = tf.keras.layers.Dense(512, activation='linear', name='dense_2')(re_lu)
        re_lu_2 = tf.keras.layers.ReLU(name='re_lu_2')(dense_2)
        dropout = tf.keras.layers.Dropout(0.25, name='dropout')(re_lu_2)

        dense_3 = tf.keras.layers.Dense(512, activation='linear', name='dense_3')(re_lu_1)
        re_lu_3 = tf.keras.layers.ReLU(name='re_lu_3')(dense_3)

        dense_4 = tf.keras.layers.Dense(32, activation='linear', name='dense_4')(dropout)
        re_lu_4 = tf.keras.layers.ReLU(name='re_lu_4')(dense_4)
        dropout_1 = tf.keras.layers.Dropout(0.25, name='dropout_1')(re_lu_4)

        dense_5 = tf.keras.layers.Dense(512, activation='linear', name='dense_5')(re_lu_3)
        re_lu_5 = tf.keras.layers.ReLU(name='re_lu_5')(dense_5)

        concatenate = tf.keras.layers.Concatenate(name='concatenate')([dropout_1, re_lu_5])

        dense_6 = tf.keras.layers.Dense(16, activation='linear', name='dense_6')(concatenate)
        re_lu_6 = tf.keras.layers.ReLU(name='re_lu_6')(dense_6)

        output_layer = tf.keras.layers.Dense(2, activation='linear', name='regression_head_1')(re_lu_6)

        model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=output_layer, name='model')

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            jit_compile=False
        )

        model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'accuracy'])

        return model


    def build_model_autokeras(self, designing:bool, overwrite:bool, tuner:str , random_seed:int, autokeras_project_name:str, auokeras_folder:str, max_trials:int = 100):
        # Entradas
        inputSensors = ak.Input(name='input_sensors')
        InputMap = ak.Input(name='input_map')

        if designing:

            # Capas ocultas para cada entrada
            hiddenLayer_sensors = ak.DenseBlock(use_batchnorm=False, name='dense_sensors')(inputSensors)
            hiddenLayer_map = ak.DenseBlock(use_batchnorm=False, name='dense_map')(InputMap)

            # Concatenamos las capas
            concat = ak.Merge()([hiddenLayer_sensors, hiddenLayer_map])

            # Capas ocultas tras la concatenación
            hiddenLayer = ak.DenseBlock(use_batchnorm=False)(concat)

        else:
            # Capas ocultas para cada entrada
            hiddenLayer_sensors = ak.DenseBlock(use_batchnorm=False, name='dense_sensors_1', num_layers=1, num_units=64)(inputSensors)
            hiddenLayer_sensors = ak.DenseBlock(use_batchnorm=False, name='dense_sensors_2', num_layers=1, num_units=512)(hiddenLayer_sensors)
            hiddenLayer_sensors = ak.DenseBlock(use_batchnorm=False, name='dense_sensors_2', num_layers=1, num_units=32)(hiddenLayer_sensors)

            hiddenLayer_map = ak.DenseBlock(use_batchnorm=False, name='dense_map', num_layers=1, num_units=16)(InputMap)
            hiddenLayer_map = ak.DenseBlock(use_batchnorm=False, name='dense_map', num_layers=1, num_units=512)(hiddenLayer_map)
            hiddenLayer_map = ak.DenseBlock(use_batchnorm=False, name='dense_map', num_layers=1, num_units=512)(hiddenLayer_map)

            # Concatenamos las capas
            concat = ak.Merge(merge_type='concatenate')([hiddenLayer_sensors, hiddenLayer_map])

            # Capas ocultas tras la concatenación
            hiddenLayer = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=16)(concat)

        # Salida
        output = ak.RegressionHead(metrics=['mse', 'accuracy'])(hiddenLayer)

        # Construimos el modelo
        model = ak.AutoModel(
                inputs=[inputSensors, InputMap],
                outputs=output, 
                overwrite=overwrite,
                tuner=tuner,
                seed=random_seed,
                max_trials=max_trials, project_name=autokeras_project_name, directory=auokeras_folder)

        return model