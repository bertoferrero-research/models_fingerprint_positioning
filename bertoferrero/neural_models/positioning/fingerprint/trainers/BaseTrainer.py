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

from sklearn.model_selection import train_test_split, GroupShuffleSplit
import tensorflow as tf
from abc import ABC, abstractmethod
import keras_tuner
import autokeras as ak
from typing import Union
import csv
import numpy as np
import pandas as pd


class BaseTrainer(ABC):
    @abstractmethod
    def train_model(dataset_path: str, scaler_file: str, tuner: str, tmp_dir: str, batch_size: int, designing: bool, overwrite: bool, max_trials: int = 100, random_seed: int = 42, hyperparams_log_path: str = None):
        pass

    @abstractmethod
    def train_model_noautoml(dataset_path: str, scaler_file: str, batch_size: int, empty_values: bool = False, random_seed: int = 42, base_model_path: str = None, disable_dropouts: bool = False):
        pass

    @abstractmethod
    def prediction(dataset_path: str, model_file: str, scaler_file: str):
        pass

    @staticmethod
    def fit_general(model, X, y, designing, batch_size, random_seed, callbacks=None, test_size: float = 0.2):

        # #region particionamos agrupando

        # # Combina las columnas 'pos_x' y 'pos_y' en una sola columna de grupos
        # # Asegúrate de que X y y sean DataFrames de pandas
        # if isinstance(X, np.ndarray):
        #     X = pd.DataFrame(X)
        # if isinstance(y, np.ndarray):
        #     y = pd.DataFrame(y)

        # # Usa todas las columnas de y para la agrupación
        # pos = y.apply(tuple, axis=1)

        # # Crea el objeto GroupShuffleSplit
        # gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)

        # # Realiza el split
        # train_idx, val_idx = next(gss.split(X, y, groups=pos))

        # # Divide los datos en conjuntos de entrenamiento y validación
        # X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        # y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # #endregion

        # Particionamos
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_seed)
        # Entrenamos
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  verbose=(1 if designing else 2), callbacks=callbacks, batch_size=batch_size, epochs=1000)

        # Evaluamos
        score = model.evaluate(X_val, y_val, verbose=0)
        return model, score

    @staticmethod
    def get_model_instance(model_file: str):
        return tf.keras.models.load_model(model_file, custom_objects=ak.CUSTOM_OBJECTS)

    @staticmethod
    def automl_trials_logger(tuner, log_file: str, num_trials: int = 10):
        """
        Logs the hyperparameters and metrics of the best trials from an AutoML tuner to a CSV file.
        Args:
            tuner: The AutoML tuner object that contains the trials.
            log_file (str): The path to the CSV file where the log will be saved.
            num_trials (int, optional): The number of best trials to log. Defaults to 10.
        Writes:
            A CSV file with the hyperparameters and scores of the best trials.
        """
        # Obtener todos los ensayos (trials) realizados
        trials = tuner.oracle.get_best_trials(num_trials=num_trials)

        # Guardar los hiperparámetros y métricas en un archivo CSV
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Escribir la cabecera (nombres de los hiperparámetros)
            writer.writerow(
                list(trials[0].hyperparameters.values.keys()) + ['score'])

            # Escribir los valores de cada ensayo
            for trial in trials:
                writer.writerow(
                    list(trial.hyperparameters.values.values()) + [trial.score])

  