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
from .BaseTrainer import BaseTrainer
import tensorflow as tf
import autokeras as ak
from bertoferrero.neural_models.positioning.fingerprint.trainingcommon import descale_numpy
import keras_tuner
import pandas as pd

class M2Trainer(BaseTrainer):
    @staticmethod
    def train_model(dataset_path: str, scaler_file: str, tuner: str, tmp_dir: str, batch_size: int, designing: bool, overwrite: bool, max_trials:int = 100, random_seed: int = 42, hyperparams_log_path: str = None):
               
        #Definimos el nombre del modelo
        modelName = 'M2'

        #Cargamos los datos de entrenamiento
        X, y = M2.load_traning_data(dataset_path, scaler_file)

        #Instanciamos la clase del modelo
        modelInstance = M2(X.shape[1], y.shape[1])

        #Construimos el modelo autokeras
        model = modelInstance.build_model_autokeras(designing=designing, overwrite=overwrite, tuner=tuner, random_seed=random_seed, autokeras_project_name=modelName, auokeras_folder=tmp_dir, max_trials=max_trials)

        # Convertir X y y a numpy.ndarray (autokeras input require de numpy.ndarray o tf.data.Dataset)
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.DataFrame) else y

        #Entrenamos
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
        model, score = BaseTrainer.fit_general(model, X_np, y_np, designing, batch_size, callbacks=[callback], random_seed=random_seed)

        #Registramos hiperparámetros
        if(hyperparams_log_path is not None):
            BaseTrainer.automl_trials_logger(model.tuner, hyperparams_log_path, max_trials)

        # Devolvemos el modelo entrenado
        model = model.export_model()

        return model, score
    
    @staticmethod
    def train_model_noautoml(dataset_path: str, scaler_file: str, batch_size: int, empty_values: bool = False, random_seed: int = 42, base_model_path: str = None):

        #Cargamos los datos de entrenamiento
        X, y = M2.load_traning_data(dataset_path, scaler_file)

        #Instanciamos la clase del modelo
        modelInstance = M2(X.shape[1], y.shape[1])

        #Construimos el modelo
        model = modelInstance.build_model(empty_values=empty_values, random_seed=random_seed, base_model_path=base_model_path)

        #Entrenamos
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
        model, score = BaseTrainer.fit_general(model, X, y, False, batch_size, callbacks=[callback], random_seed=random_seed)

        return model, score
    
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


        return BaseTrainer.base_fine_tuning(
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

    @staticmethod
    def prediction(dataset_path: str, model_file: str, scaler_file: str):
        #Cargamos los datos de entrenamiento
        input_data, output_data = M2.load_testing_data(dataset_path, scaler_file)
        output_data = output_data.to_numpy()

        #Cargamos el modelo
        model = tf.keras.models.load_model(model_file, custom_objects=ak.CUSTOM_OBJECTS)

        #Evaluamos
        metrics = model.evaluate(input_data, output_data, verbose=0)

        #Predecimos
        predictions = model.predict(input_data)

        #Los datos de predicción y salida vienen escalados, debemos desescalarlos
        output_data = descale_numpy(output_data)
        predictions = descale_numpy(predictions)

        #Formateamos las métricas
        formated_metrics = {
            'loss_mse': metrics[1],
            'accuracy': metrics[2]
        }

        #Devolvemos las predicciones y los datos de salida esperados
        return predictions, output_data, formated_metrics