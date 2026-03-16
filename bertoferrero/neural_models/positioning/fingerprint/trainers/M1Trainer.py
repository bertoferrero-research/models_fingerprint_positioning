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

from bertoferrero.neural_models.positioning.fingerprint.models import M1
from bertoferrero.neural_models.positioning.fingerprint.trainingcommon import descale_numpy
from sklearn.model_selection import train_test_split
from .BaseTrainer import BaseTrainer
import tensorflow as tf
import autokeras as ak
import pandas as pd

class M1Trainer(BaseTrainer):
    @staticmethod
    def train_model(dataset_path: str, scaler_file: str, tuner: str, tmp_dir: str, batch_size: int, designing: bool, overwrite: bool, max_trials:int = 100, random_seed: int = 42, hyperparams_log_path: str = None, pos_limits: dict = None):
        """
        Entrena el modelo M1 usando AutoKeras.

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
        modelName = 'M1'

        #Cargamos los datos de entrenamiento
        X, y = M1.load_traning_data(dataset_path, scaler_file, pos_limits)

        #Instanciamos la clase del modelo
        modelInstance = M1(X.shape[1], y.shape[1])

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
    def prediction(dataset_path: str, model_file: str, scaler_file: str, pos_limits: dict = None):
        """
        Genera predicciones con el modelo M1 entrenado y evalúa su rendimiento.

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
        input_data, output_data = M1.load_testing_data(dataset_path, scaler_file, pos_limits)
        output_data = output_data.to_numpy()

        #Cargamos el modelo
        model = tf.keras.models.load_model(model_file, custom_objects=ak.CUSTOM_OBJECTS)

        #Evaluamos
        metrics = model.evaluate(input_data, output_data, verbose=0)

        #Predecimos
        predictions = model.predict(input_data)

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

