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

class M7(ModelsBaseClass): 
    @staticmethod
    def load_traning_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, train_scaler_file=True, include_pos_z=False, scale_y=False)

    @staticmethod
    def load_testing_data(data_file: str, scaler_file: str):
        return load_data(data_file, scaler_file, train_scaler_file=False, include_pos_z=False, scale_y=False)

    def build_model(self):
        input = tf.keras.layers.Input(shape=(self.inputlength,)) 
        layer = tf.keras.layers.Dense(16, activation='relu')(input)
        layer = tf.keras.layers.Dense(32, activation='relu')(layer)
        output = tf.keras.layers.Dense(42, activation='softmax')(layer)

        model = tf.keras.models.Model(inputs=input, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['mse', 'accuracy'] )

        return model

    def build_model_autokeras(self, designing:bool, overwrite:bool, tuner:str , random_seed:int, autokeras_project_name:str, auokeras_folder:str, max_trials:int = 100):
        input = ak.Input()
        if designing:
            layer = ak.DenseBlock(use_batchnorm=False)(input)
        else:
            layer = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=16)(input)
            layer = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=32)(layer)
            layer = ak.DenseBlock(use_batchnorm=False, num_layers=1, num_units=42)(layer)
        output_layer = ak.ClassificationHead(metrics=['mse', 'accuracy'])(layer)

        model = ak.AutoModel(
            inputs=input,
            outputs=output_layer,
            overwrite=overwrite,
            objective = 'val_accuracy',
            tuner=tuner,
            seed=random_seed,
            max_trials=max_trials, project_name=autokeras_project_name, directory=auokeras_folder
        )
        return model