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

class M3(ModelsBaseClass): 
    @staticmethod
    def load_traning_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, train_scaler_file=True, include_pos_z=False,
                        scale_y=True, not_valid_sensor_value=100, return_valid_sensors_map=True)

    @staticmethod
    def load_testing_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, include_pos_z=False, scale_y=True, not_valid_sensor_value=100, return_valid_sensors_map=True)

    def build_model(self, empty_values: bool = False):
        pass

    def build_model_autokeras(self, designing:bool, overwrite:bool, tuner:str , random_seed:int, autokeras_project_name:str, auokeras_folder:str, max_trials:int = 100):
        # Entradas
        inputSensors = ak.Input()
        InputMap = ak.Input()

        # Concatenamos las capas
        concat = ak.Merge(merge_type='concatenate')([inputSensors, InputMap])

        # Capas ocultas tras la concatenación
        #Para el diseñado
        if designing:
            hiddenLayer = ak.DenseBlock(use_batchnorm=False)(concat)
        else:
            hiddenLayer = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=1024)(concat)
            hiddenLayer = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=128)(hiddenLayer)

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