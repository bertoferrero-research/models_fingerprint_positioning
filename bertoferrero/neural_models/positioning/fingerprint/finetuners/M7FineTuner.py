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

from bertoferrero.neural_models.positioning.fingerprint.models import M7
from .BaseFineTuner import BaseFineTuner
import tensorflow as tf
from bertoferrero.neural_models.positioning.fingerprint.trainingcommon import posXYlist_to_grid, gridList_to_posXY
import keras_tuner

class M7FineTuner(BaseFineTuner):    
    
    @staticmethod
    def fine_tuning(model_file: str, dataset_path: str, scaler_file: str, tmp_dir: str, batch_size: int, overwrite: bool, max_trials:int = 100, random_seed: int = 42, hyperparams_log_path: str = None):
            
        modelName = 'M7'
        training_loss = 'categorical_crossentropy'
        training_optimizer = 'sgd'
        training_metrics = ['mse', 'accuracy']
        training_learning_rate=0.1
        cell_amount_x = 7
        cell_amount_y = 6
        
        #Preparamos datos de entrenamiento
        X, y = M7.load_testing_data(dataset_path, scaler_file)

        #Convertimos a categorical
        y = posXYlist_to_grid(y.to_numpy(), cell_amount_x, cell_amount_y)
        y = tf.keras.utils.to_categorical(y, num_classes=cell_amount_x*cell_amount_y)

        #Definimos el rango de la tasa de aprendizaje inicial
        learning_rate_max = training_learning_rate
        learning_rate_min = learning_rate_max / 100

        # Preparamos Callbacks
        def build_callbacks_fn(hp):
            callback_early = tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', min_delta=0.0001, patience=10, restore_best_weights=True)
            callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy', factor=0.2, patience=5, min_lr=learning_rate_min)
            return [callback_early, callback_reduce_lr]

        return BaseFineTuner.base_fine_tuning(
            modelName=modelName,
            X=X,
            y=y,
            hyperModel=None,
            tunerObjectives=keras_tuner.Objective("val_accuracy", direction="max"),
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
            random_seed=random_seed,
            hyperparams_log_path=hyperparams_log_path,
            build_callbacks_fn=build_callbacks_fn
        )

