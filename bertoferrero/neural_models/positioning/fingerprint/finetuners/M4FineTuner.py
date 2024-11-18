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

from bertoferrero.neural_models.positioning.fingerprint.models import M4
from bertoferrero.neural_models.positioning.fingerprint.trainingcommon import set_random_seed_value
from bertoferrero.neural_models.positioning.fingerprint.trainingcommon import plot_learning_curves
from .BaseFineTuner import BaseFineTuner
import tensorflow as tf
import pandas as pd
import keras_tuner
import numpy as np
import autokeras as ak
from itertools import chain
from sklearn.model_selection import train_test_split

class M4FineTuner(BaseFineTuner):
    
    @staticmethod
    def load_data_and_model(dataset_path, scaler_file, model_file):
        X, y, Xmap = M4.load_traning_data(dataset_path, scaler_file)
        model = tf.keras.models.load_model(model_file, custom_objects=ak.CUSTOM_OBJECTS)
        return [X, Xmap], y, model

    @staticmethod
    def prepare_callbacks(learning_rate_min):
        callback_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
        callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=learning_rate_min)
        return [callback_early, callback_reduce_lr]

    @staticmethod
    def freeze_layers(model, m4_layers_data_input, m4_layers_mask_input, layers_to_freeze_data_input, layers_to_freeze_mask_input):
        layers_name_to_freeze = list(chain(*(m4_layers_data_input[:layers_to_freeze_data_input]), *(m4_layers_mask_input[:layers_to_freeze_mask_input])))
        BaseFineTuner.freeze_layers(model, layers_name_to_freeze)

    @staticmethod
    def get_layers_definition():
        return [
            ['input_1'],                            # Entrada
            ['dense', 're_lu'],                     # Capa 64
            ['dense_2', 're_lu_2', 'dropout'],      # Capa 512
            ['dense_4', 're_lu_4', 'dropout_1']     # Capa 32
        ], [
            ['input_2'],                            # Entrada
            ['dense_1', 're_lu_1'],                 # Capa 16
            ['dense_3', 're_lu_3'],                 # Capa 512
            ['dense_5', 're_lu_5']                  # Capa 512
        ]

    @staticmethod
    def fine_tuning(model_file: str, dataset_path: str, scaler_file: str, tmp_dir: str, batch_size: int, overwrite: bool, max_trials:int = 100, random_seed: int = 42, hyperparams_log_path: str = None):
            
        modelName = 'M4'
        training_loss = 'mse'
        training_metrics = ['mse', 'accuracy']
        training_learning_rate=0.001
        
        # Preparamos datos de entrenamiento
        X, y, base_model = M4FineTuner.load_data_and_model(dataset_path, scaler_file, model_file)

        # Definimos el rango de la tasa de aprendizaje inicial
        learning_rate_max = training_learning_rate
        learning_rate_min = learning_rate_max / 100

        # Preparamos Callbacks
        callbacks = M4FineTuner.prepare_callbacks(learning_rate_min)

        # Definimos los grupos de capas para la desactivación
        m4_layers_data_input, m4_layers_mask_input = M4FineTuner.get_layers_definition()

        # Preparamos el hypermodelo personalizado
        def M4HyperModel(hp):
            # Clonamos el modelo base para que en cada iteración se empiece desde el mismo punto
            model = tf.keras.models.clone_model(base_model)

            # Cargamos los pesos del modelo base
            model.set_weights(base_model.get_weights())

            # Definimos el conjunto de capas a congelar por cada grupo
            layers_to_freeze_data_input = hp.Int(
                'layers_to_freeze_data_input', min_value=0, max_value=len(m4_layers_data_input))
            layers_to_freeze_mask_input = hp.Int(
                'layers_to_freeze_mask_input', min_value=0, max_value=len(m4_layers_mask_input))

            # Congelamos las capas que tocan
            M4FineTuner.freeze_layers(model, m4_layers_data_input, m4_layers_mask_input, layers_to_freeze_data_input, layers_to_freeze_mask_input)

            # Definimos la tasa de aprendizaje
            learning_rate = hp.Float('learning_rate', min_value=learning_rate_min,
                                        max_value=learning_rate_max, step=10, sampling="log")

            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate)

            # Compilamos el modelo
            model.compile(
                loss=training_loss,
                optimizer=optimizer,
                metrics=training_metrics
            )

            return model

        return BaseFineTuner.base_fine_tuning(
            modelName=modelName,
            X=X,
            y=y,
            hyperModel=M4HyperModel,
            tunerObjectives=keras_tuner.Objective("val_loss", direction="min"),
            tmp_dir=tmp_dir,
            batch_size=batch_size,
            overwrite=overwrite,
            model_file=model_file,
            learning_rate_max=learning_rate_max,
            learning_rate_min=learning_rate_min,
            training_loss=training_loss,
            training_metrics=training_metrics,
            max_trials=max_trials,
            random_seed=random_seed,
            hyperparams_log_path=hyperparams_log_path,
            callbacks=callbacks
        )

    @staticmethod
    def fine_tuning_noautoml(model_file: str, dataset_path: str, scaler_file: str, batch_size: int, random_seed: int = 42):
        training_loss = 'mse'
        training_metrics = ['mse', 'accuracy']
        training_learning_rate = 0.001
        layers_to_freeze_data_input = 0
        layers_to_freeze_mask_input = 2
        
        set_random_seed_value(seed=random_seed)

        # Preparamos datos de entrenamiento
        X, y, model = M4FineTuner.load_data_and_model(dataset_path, scaler_file, model_file)
        splited_data = train_test_split(*X, y, test_size=0.2, random_state=random_seed)            
        y_train, y_val = splited_data[-2], splited_data[-1]
        X_train = splited_data[:-2][0::2]
        X_val = splited_data[:-2][1::2]

        # Definimos los grupos de capas para la desactivación
        m4_layers_data_input, m4_layers_mask_input = M4FineTuner.get_layers_definition()
        
        # Congelamos las capas que tocan
        M4FineTuner.freeze_layers(model, m4_layers_data_input, m4_layers_mask_input, layers_to_freeze_data_input, layers_to_freeze_mask_input)

        # Compilamos el modelo
        model.compile(
            loss=training_loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=training_learning_rate),
            metrics=training_metrics
        )

        # Preparamos callbacks
        callbacks = M4FineTuner.prepare_callbacks(training_learning_rate/100)

        # Entrenamos el modelo
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, batch_size=batch_size, callbacks=callbacks, verbose=2)


        # Evaluamos el modelo
        score = model.evaluate(X_val, y_val, verbose=0)

        return model, score, history

