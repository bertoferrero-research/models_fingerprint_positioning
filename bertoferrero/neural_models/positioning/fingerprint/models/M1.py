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

class M1(ModelsBaseClass):

    @staticmethod
    def load_traning_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, train_scaler_file=True, include_pos_z=False, scale_y=True)

    @staticmethod
    def load_testing_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, train_scaler_file=False, include_pos_z=False, scale_y=True)

    def build_model(self, empty_values: bool = False):
        input = tf.keras.layers.Input(shape=self.inputlength)        
        hiddenLayerLength = round(self.inputlength*2/3+self.outputlength, 0)
        hiddenLayer = tf.keras.layers.Dense(hiddenLayerLength, activation='relu')(input)
        output = tf.keras.layers.Dense(self.outputlength, activation='linear')(hiddenLayer)

        model = tf.keras.models.Model(inputs=input, outputs=output)
        model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'] )

        return model

    #Faltaría aquí un build_model_autokeras pero con M1 no es necesario
    def build_model_autokeras(self, designing:bool, overwrite:bool, tuner:str , random_seed:int, autokeras_project_name:str, auokeras_folder:str, max_trials:int = 100):
        input = ak.Input()
        hiddenLayerLength = round(self.inputlength*2/3+self.outputlength, 0)
        hiddenLayers = ak.DenseBlock(num_layers=1, num_units=hiddenLayerLength, use_batchnorm=False)(input)
        output = ak.RegressionHead(metrics=['mse', 'accuracy'])(hiddenLayers)

        model = ak.AutoModel(
            inputs=input,
            outputs=output,
            overwrite=overwrite,
            seed=random_seed,
            max_trials=max_trials, project_name=autokeras_project_name, directory=auokeras_folder,
            tuner=tuner
        )

        return model