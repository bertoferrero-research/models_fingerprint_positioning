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

from bertoferrero.neural_models.positioning.fingerprint.models import M2
from sklearn.model_selection import train_test_split
from .BaseFineTuner import BaseFineTuner
import tensorflow as tf
import autokeras as ak
from bertoferrero.neural_models.positioning.fingerprint.trainingcommon import descale_numpy
import keras_tuner
import pandas as pd
from itertools import chain

class M2FineTuner(BaseFineTuner):
    
    
    @staticmethod
    def fine_tuning(model_file: str, dataset_path: str, scaler_file: str, tmp_dir: str, batch_size: int, overwrite: bool, max_trials:int = 100, random_seed: int = 42, hyperparams_log_path: str = None):
            
        modelName = 'M2'
        training_loss = 'mse'
        training_optimizer = 'adam'
        training_metrics = ['mse', 'accuracy']
        training_learning_rate=0.0001
        
        #Preparamos datos de entrenamiento
        X, y = M2.load_testing_data(dataset_path, scaler_file)

        #Definimos el rango de la tasa de aprendizaje inicial
        learning_rate_max = training_learning_rate
        learning_rate_min = learning_rate_max / 100

        
        # Preparamos Callbacks
        callback_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
        callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr = learning_rate_min)
        callbacks = [callback_early, callback_reduce_lr]

        # Cargamos el modelo
        base_model = tf.keras.models.load_model(
            model_file, custom_objects=ak.CUSTOM_OBJECTS)
        
        # Definimos los grupos de capas para la desactivación
        m2_layers = [
            ['input_1'],                            #Entrada
            ['dense', 're_lu', 'dropout'],          #Capa 1024
            ['dense_1', 're_lu_1'],                 #Capa 128
            ['dense_2', 're_lu_2', 'dropout_1'],    #Capa 16
        ]

        #Preparamos el hypermodelo personalizado
        def M2HyperModel(hp):
            # Clonamos el modelo base para que en cada iteración se empiece desde el mismo punto
            model = tf.keras.models.clone_model(base_model)

            # Cargamos los pesos del modelo base
            model.set_weights(base_model.get_weights())

            # Definimos el conjunto de capas a congelar por cada grupo
            groups_layers_to_freeze = hp.Int(
                'groups_layers_to_freeze', min_value=0, max_value=len(m2_layers))

            # Congelamos las capas que tocan
            layers_name_to_freeze = list(chain(*(m2_layers[:groups_layers_to_freeze])))
            layers_to_freeze = [layer for layer in model.layers if layer.name in layers_name_to_freeze]
            for layer in layers_to_freeze:
                layer.trainable = False

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
            hyperModel=M2HyperModel,
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

    