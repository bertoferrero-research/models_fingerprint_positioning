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

from sklearn.model_selection import train_test_split
import tensorflow as tf
from abc import ABC, abstractmethod
import keras_tuner
import autokeras as ak
from typing import Union

class BaseTrainer(ABC):
    @abstractmethod
    def train_model(dataset_path: str, scaler_file: str, tuner: str, tmp_dir: str, batch_size: int, designing: bool, overwrite: bool, max_trials:int = 100, random_seed: int = 42):
        pass

    @abstractmethod
    def prediction(dataset_path: str, model_file: str, scaler_file: str):
        pass
    
    @staticmethod
    def fit_autokeras(model, X, y, designing, batch_size, callbacks = None, test_size:float=0.2):
        #Particionamos
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size)
        #Entrenamos
        model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                            verbose=(1 if designing else 2), callbacks=callbacks, batch_size=batch_size)
        #Evaluamos
        score = model.evaluate(X_val, y_val, verbose=0)
        return model, score
    
    