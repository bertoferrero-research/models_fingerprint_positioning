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
from sklearn.model_selection import train_test_split
from .BaseTrainer import BaseTrainer
import tensorflow as tf
import numpy as np
import autokeras as ak
from bertoferrero.neural_models.positioning.fingerprint.trainingcommon import descale_numpy
import pandas as pd

class M4Trainer(BaseTrainer):
    @staticmethod
    def train_model(dataset_path: str, scaler_file: str, tuner: str, tmp_dir: str, batch_size: int, designing: bool, overwrite: bool, max_trials:int = 100, random_seed: int = 42, hyperparams_log_path: str = None, pos_limits: dict = None):
        """
        Entrena el modelo M4 usando AutoKeras con doble entrada (RSSI + mapa de sensores).

        Args:
            dataset_path (str): Ruta al fichero CSV de datos de entrenamiento.
            scaler_file (str): Ruta del fichero scaler para RSSI.
            tuner (str): Tipo de tuner AutoKeras ('bayesian', 'hyperband', 'random').
            tmp_dir (str): Directorio temporal para los artefactos de AutoKeras.
            batch_size (int): Tamaño del batch de entrenamiento.
            designing (bool): Si es True, modo diseño con búsqueda libre de hiperparámetros.
            overwrite (bool): Si es True, sobreescribe búsquedas previas de AutoKeras.
            max_trials (int, optional): Número máximo de trials para AutoKeras.
            random_seed (int, optional): Semilla aleatoria para reproducibilidad.
            hyperparams_log_path (str, optional): Ruta para registrar los hiperparámetros encontrados.
            pos_limits (dict, optional): Límites de posición {'min_x', 'max_x', 'min_y', 'max_y'}.

        Returns:
            tuple: (model, score) con el modelo Keras exportado y la puntuación de evaluación.
        """
               
        #Definimos el nombre del modelo
        modelName = 'M4'

        #Cargamos los datos de entrenamiento
        X, y, Xmap = M4.load_traning_data(dataset_path, scaler_file, pos_limits)

        #Convertimos a numpy y formato
        X = X.to_numpy()
        y = y.to_numpy()
        Xmap = Xmap.to_numpy()

        #Instanciamos la clase del modelo
        modelInstance = M4(X.shape[1], y.shape[1])

        #Construimos el modelo autokeras
        model = modelInstance.build_model_autokeras(designing=designing, overwrite=overwrite, tuner=tuner, random_seed=random_seed, autokeras_project_name=modelName, auokeras_folder=tmp_dir, max_trials=max_trials)
        
        # Convertir X y y a numpy.ndarray (autokeras input require de numpy.ndarray o tf.data.Dataset)
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.DataFrame) else y

        # Entrenamos
        X_train, X_val, y_train, y_val, Xmap_train, Xmap_val = train_test_split(
            X_np, y_np, Xmap, test_size=0.2)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
        history = model.fit([X_train, Xmap_train], y_train, validation_data=([X_val, Xmap_val], y_val),
                            verbose=(1 if designing else 2), callbacks=[callback], batch_size=batch_size)

        #Registramos hiperparámetros
        if(hyperparams_log_path is not None):
            BaseTrainer.automl_trials_logger(model.tuner, hyperparams_log_path, max_trials)

        # Evaluamos usando el test set
        score = model.evaluate([X_val, Xmap_val], y_val, verbose=0)

        # Devolvemos el modelo entrenado
        model = model.export_model()

        return model, score
    
    @staticmethod
    def train_model_noautoml(
        dataset_path: str,
        scaler_file: str,
        batch_size: int,
        empty_values: bool = False,
        random_seed: int = 42,
        base_model_path: str = None,
        disable_dropouts: bool = False,
        pos_limits: dict = None,
        sample_weight = None):        
        """
        Entrena el modelo M4 sin AutoKeras, usando la arquitectura fija con doble entrada
        (RSSI + mapa de sensores).

        Args:
            dataset_path (str): Ruta al fichero CSV de datos de entrenamiento.
            scaler_file (str): Ruta del fichero scaler para RSSI.
            batch_size (int): Tamaño del batch de entrenamiento.
            empty_values (bool, optional): Si es True, adapta el modelo para datos con valores faltantes.
            random_seed (int, optional): Semilla aleatoria para reproducibilidad.
            base_model_path (str, optional): Ruta a un modelo base para transfer learning.
            disable_dropouts (bool, optional): Si es True, desactiva las capas dropout.
            pos_limits (dict, optional): Límites de posición {'min_x', 'max_x', 'min_y', 'max_y'}.

        Returns:
            tuple: (model, score, history) con el modelo entrenado, puntuación de evaluación
                e historial de entrenamiento.
        """

        #Cargamos los datos de entrenamiento
        X, y, Xmap = M4.load_traning_data(dataset_path, scaler_file, pos_limits)

        #Convertimos a numpy y formato
        X = X.to_numpy()
        y = y.to_numpy()
        Xmap = Xmap.to_numpy()

        #Instanciamos la clase del modelo
        modelInstance = M4(X.shape[1], y.shape[1])

        #Construimos el modelo
        model = modelInstance.build_model(empty_values=empty_values, random_seed=random_seed, base_model_path=base_model_path, disable_dropouts=disable_dropouts)

        # Entrenamos
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
        X_train, X_val, y_train, y_val, Xmap_train, Xmap_val = train_test_split(
            X, y, Xmap, test_size=0.2, random_state=random_seed)
        
        history = model.fit([X_train, Xmap_train], y_train, validation_data=([X_val, Xmap_val], y_val),
                  verbose=2, callbacks=[callback], batch_size=batch_size, epochs=1000)
        score = model.evaluate([X_val, Xmap_val], y_val, verbose=0)

        return model, score, history

    @staticmethod
    def prediction(dataset_path: str, model_file: str, scaler_file: str, pos_limits: dict = None):
        """
        Genera predicciones con el modelo M4 entrenado y evalúa su rendimiento.
        El modelo acepta doble entrada: datos RSSI y mapa de sensores válidos.

        Args:
            dataset_path (str): Ruta al fichero CSV de datos de evaluación.
            model_file (str): Ruta al fichero del modelo Keras guardado.
            scaler_file (str): Ruta del fichero scaler para RSSI.
            pos_limits (dict, optional): Límites de posición {'min_x', 'max_x', 'min_y', 'max_y'}.

        Returns:
            tuple: (predictions, output_data, formated_metrics) donde predictions y output_data
                son arrays desescalados y formated_metrics contiene 'loss_mse' y 'accuracy'.
        """
        #Cargamos los datos de entrenamiento
        input_data, output_data, input_map_data = M4.load_testing_data(dataset_path, scaler_file, pos_limits)
        output_data = output_data.to_numpy()

        #Cargamos el modelo
        model = tf.keras.models.load_model(model_file, custom_objects=ak.CUSTOM_OBJECTS)
        
        #Evaluamos
        metrics = model.evaluate([input_data, input_map_data], output_data, verbose=0)

        #Predecimos
        predictions = model.predict([input_data, input_map_data])

        #Los datos de predicción y salida vienen escalados, debemos desescalarlos
        output_data = descale_numpy(output_data, pos_limits)
        predictions = descale_numpy(predictions, pos_limits)

        #Formateamos las métricas
        formated_metrics = {
            'loss_mse': metrics[1],
            'accuracy': metrics[2]
        }

        #Devolvemos las predicciones y los datos de salida esperados
        return predictions, output_data, formated_metrics