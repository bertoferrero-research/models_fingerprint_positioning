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

from bertoferrero.neural_models.positioning.fingerprint.models import M5
from sklearn.model_selection import train_test_split
from .BaseTrainer import BaseTrainer
import tensorflow as tf
import numpy as np
import autokeras as ak
from bertoferrero.neural_models.positioning.fingerprint.trainingcommon import descale_numpy
import pandas as pd

class M5Trainer(BaseTrainer):
    @staticmethod
    def train_model(dataset_path: str, scaler_file: str, tuner: str, tmp_dir: str, batch_size: int, designing: bool, overwrite: bool, max_trials:int = 100, random_seed: int = 42, hyperparams_log_path: str = None):
               
        #Definimos el nombre del modelo
        modelName = 'M5'

        #Cargamos los datos de entrenamiento
        X, y = M5.load_traning_data(dataset_path, scaler_file)

        #Cambiamos dimensionalidad para que sea compatible con CNN1D
        X = X.values.reshape(X.shape[0], X.shape[1], 1)
        #X = np.expand_dims(X, axis=2) #Otra forma de hacerlo

        #Instanciamos la clase del modelo
        modelInstance = M5(X.shape[1], y.shape[1])

        #Construimos el modelo autokeras
        model = modelInstance.build_model_autokeras(designing=designing, overwrite=overwrite, tuner=tuner, random_seed=random_seed, autokeras_project_name=modelName, auokeras_folder=tmp_dir, max_trials=max_trials)

        #Entrenamos
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
        model, score = BaseTrainer.fit_general(model, X, y, designing, batch_size, callbacks=[callback], random_seed=random_seed)

        #Registramos hiperparámetros
        if(hyperparams_log_path is not None):
            BaseTrainer.automl_trials_logger(model.tuner, hyperparams_log_path, max_trials)

        # Devolvemos el modelo entrenado
        model = model.export_model()

        return model, score

    @staticmethod
    def prediction(dataset_path: str, model_file: str, scaler_file: str):
        #Cargamos los datos de entrenamiento
        input_data, output_data = M5.load_testing_data(dataset_path, scaler_file)
        input_data = input_data.values.reshape(input_data.shape[0], input_data.shape[1], 1)
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
