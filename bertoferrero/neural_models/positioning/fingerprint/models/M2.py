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
import autokeras as ak
from bertoferrero.neural_models.positioning.fingerprint.trainingcommon import load_data
from .ModelsBaseClass import ModelsBaseClass

class M2(ModelsBaseClass): 
    @staticmethod
    def load_traning_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, train_scaler_file=True, include_pos_z=False, scale_y=True)

    @staticmethod
    def load_testing_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, train_scaler_file=False, include_pos_z=False, scale_y=True)

    def build_model(self, random_seed:int, empty_values: bool = False, base_model_path: str = None):
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)

        #Cargamos el modelo partiendo de una base o integramente desde la api
        if base_model_path is not None:
            return self._build_model_from_base(base_model_path=base_model_path, loss='mse', metrics=['mse', 'accuracy'])

        input = tf.keras.layers.Input(shape=(self.inputlength,))     
        layers = tf.keras.layers.Dense(1024, activation='relu')(input)
        if not empty_values:
            layers = tf.keras.layers.Dropout(0.25)(layers)
        else:
            layers = tf.keras.layers.Dropout(0.5)(layers)

        layers = tf.keras.layers.Dense(128, activation='relu')(layers)
        layers = tf.keras.layers.Dense(16, activation='relu')(layers)
        if empty_values:
            layers = tf.keras.layers.Dropout(0.25)(layers)
            
        output = tf.keras.layers.Dense(self.outputlength, activation='linear')(layers)

        model = tf.keras.models.Model(inputs=input, outputs=output)
        if not empty_values:
            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['mse', 'accuracy'] )
        else:
            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['mse', 'accuracy'] )

        return model

    def build_model_autokeras(self, designing:bool, overwrite:bool, tuner:str , random_seed:int, autokeras_project_name:str, auokeras_folder:str, max_trials:int = 100):
        input = ak.Input()
        if designing:
            hiddenLayers = ak.DenseBlock(use_batchnorm=False)(input)
        else:
            hiddenLayers = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=1024)(input)
            hiddenLayers = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=128)(hiddenLayers)
            hiddenLayers = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=16)(hiddenLayers)
        
        output = ak.RegressionHead(output_dim=self.outputlength, metrics=['mse', 'accuracy'])(hiddenLayers)

        model = ak.AutoModel(
            inputs=input,
            outputs=output,
            overwrite=overwrite,
            seed=random_seed,
            max_trials=max_trials, project_name=autokeras_project_name, directory=auokeras_folder,
            tuner=tuner
        )

        return model