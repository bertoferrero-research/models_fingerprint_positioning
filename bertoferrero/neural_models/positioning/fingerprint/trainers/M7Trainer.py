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

from sklearn.metrics import accuracy_score
from bertoferrero.neural_models.positioning.fingerprint.models import M7
from sklearn.model_selection import train_test_split
from .BaseTrainer import BaseTrainer
import tensorflow as tf
import numpy as np
from bertoferrero.neural_models.positioning.fingerprint.trainingcommon import posXYlist_to_grid, gridList_to_posXY
import autokeras as ak
from bertoferrero.neural_models.positioning.fingerprint.trainingcommon import descale_numpy
import pandas as pd
import keras_tuner

class M7Trainer(BaseTrainer):
    @staticmethod
    def train_model(dataset_path: str, scaler_file: str, tuner: str, tmp_dir: str, batch_size: int, designing: bool, overwrite: bool, max_trials:int = 100, random_seed: int = 42, hyperparams_log_path: str = None):
               
        #Definimos el nombre del modelo y la configuración específica
        modelName = 'M7'
        cell_amount_x = 7
        cell_amount_y = 6

        #Cargamos los datos de entrenamiento
        X, y = M7.load_traning_data(dataset_path, scaler_file)

        y = posXYlist_to_grid(y.to_numpy(), cell_amount_x, cell_amount_y)

        #Convertimos a categorical
        y = tf.keras.utils.to_categorical(y, num_classes=cell_amount_x*cell_amount_y)
        
        #Instanciamos la clase del modelo
        modelInstance = M7(X.shape[1], y.shape[1])

        #Construimos el modelo autokeras
        model = modelInstance.build_model_autokeras(designing=designing, overwrite=overwrite, tuner=tuner, random_seed=random_seed, autokeras_project_name=modelName, auokeras_folder=tmp_dir, max_trials=max_trials)
        
        # Convertir X y y a numpy.ndarray (autokeras input require de numpy.ndarray o tf.data.Dataset)
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.DataFrame) else y

        #Entrenamos
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10, restore_best_weights=True)
        model, score = BaseTrainer.fit_general(model, X_np, y_np, designing, batch_size, callbacks=[callback])

        #Registramos hiperparámetros
        if(hyperparams_log_path is not None):
            BaseTrainer.automl_trials_logger(model.tuner, hyperparams_log_path, max_trials)

        # Devolvemos el modelo entrenado
        model = model.export_model()

        return model, score
    
    @staticmethod
    def train_model_noautoml(dataset_path: str, scaler_file: str, batch_size: int):

        cell_amount_x = 7
        cell_amount_y = 6

        #Cargamos los datos de entrenamiento
        X, y = M7.load_traning_data(dataset_path, scaler_file)
        y = posXYlist_to_grid(y.to_numpy(), cell_amount_x, cell_amount_y)
        #Convertimos a categorical
        y = tf.keras.utils.to_categorical(y, num_classes=cell_amount_x*cell_amount_y)

        #Instanciamos la clase del modelo
        modelInstance = M7(X.shape[1], y.shape[1])

        #Construimos el modelo
        model = modelInstance.build_model()

        #Entrenamos
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10, restore_best_weights=True)
        model, score = BaseTrainer.fit_general(model, X, y, False, batch_size, callbacks=[callback])

        return model, score
    
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

        #Preparamos Callbacks
        callback_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10, restore_best_weights=True)
        callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr = learning_rate_min)


        return BaseTrainer.base_fine_tuning(
            modelName=modelName,
            X=X,
            y=y,
            hyperModel=None,
            tunerObjectives=keras_tuner.Objective("val_accuracy", direction="max"),
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

    @staticmethod
    def prediction(dataset_path: str, model_file: str, scaler_file: str):
        cell_amount_x = 7
        cell_amount_y = 6

        #Cargamos los datos de entrenamiento
        input_data, output_data = M7.load_testing_data(dataset_path, scaler_file)
        output_data = output_data.to_numpy()

        #Cargamos el modelo
        model = tf.keras.models.load_model(model_file, custom_objects=ak.CUSTOM_OBJECTS)

        #Predecimos
        predictions = model.predict(input_data)
        predictions = np.argmax(predictions, axis=-1)
        # Convertimos a posiciones
        predictions_positions = gridList_to_posXY(
            predictions, cell_amount_x, cell_amount_y)
        
        # Evaluación
        output_data_grid = posXYlist_to_grid(output_data, cell_amount_x, cell_amount_y)
        output_data_categorical = tf.keras.utils.to_categorical(output_data_grid, num_classes=cell_amount_x*cell_amount_y)
        #accuracy = accuracy_score(output_data_grid, predictions)
        metrics = model.evaluate(input_data, output_data_categorical, verbose=0)
        formated_metrics = {
            'loss_mse': metrics[1],
            'accuracy': metrics[2]
        }

        #Devolvemos las predicciones y los datos de salida esperados
        return predictions_positions, output_data, formated_metrics