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

        #Preparamos Callbacks
        callback_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
        callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr = learning_rate_min)


        return BaseFineTuner.base_fine_tuning(
            modelName=modelName,
            X=X,
            y=y,
            hyperModel=None,
            tunerObjectives=keras_tuner.Objective("val_loss", direction="min"),
            callbacks=[callback_early, callback_reduce_lr],
            tmp_dir=tmp_dir,
            batch_size=batch_size,
            overwrite=overwrite,
            model_file=model_file,
            learning_rate_max=learning_rate_max,
            learning_rate_min=learning_rate_min,
            training_optimizer=training_optimizer,
            training_loss=training_loss,
            training_metrics=training_metrics,
            max_trials=max_trials,
            random_seed=random_seed
        )

    