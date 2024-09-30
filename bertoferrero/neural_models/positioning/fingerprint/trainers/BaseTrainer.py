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

from sklearn.model_selection import train_test_split
import tensorflow as tf
from abc import ABC, abstractmethod
import keras_tuner
import autokeras as ak
from typing import Union

class BaseTrainer(ABC):
    @abstractmethod
    def train_model(dataset_path: str, scaler_file: str, tuner: str, tmp_dir: str, batch_size: int, designing: bool, overwrite: bool, max_trials:int = 100, random_seed: int = 42):
        pass

    @abstractmethod
    def prediction(dataset_path: str, model_file: str, scaler_file: str):
        pass
    
    @staticmethod
    def fit_autokeras(model, X, y, designing, batch_size, callbacks = None, test_size:float=0.2):
        #Particionamos
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size)
        #Entrenamos
        model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                            verbose=(1 if designing else 2), callbacks=callbacks, batch_size=batch_size)
        #Evaluamos
        score = model.evaluate(X_val, y_val, verbose=0)
        return model, score
    
    @staticmethod
    def base_fine_tuning(modelName: str,X: any, y: any, hyperModel: Union[str, None], tunerObjectives: any, callbacks:list, tmp_dir: str, batch_size: int, overwrite: bool, model_file: Union[str, None]=None, learning_rate_max: Union[float, None]=None, learning_rate_min: Union[float, None]=None, training_optimizer: Union[str, None]=None, training_loss: Union[str, None]=None, training_metrics: Union[list, None]=None, test_size:float=0.2, max_trials:int = 100, random_seed: int = 42):
        """
        Fine-tunes a base model using hyperparameter tuning.
        Args:
            modelName (str): The name of the model.
            X (any): The input data for training.
            y (any): The target data for training.
            hyperModel (Union[str, None]): The hypermodel to use for tuning. If None, the hyperparameters must be defined individually.
            tunerObjectives (any): The objectives to optimize during tuning.
            callbacks (list): List of Keras callbacks to apply during training.
            tmp_dir (str): The directory to store the tuning results.
            batch_size (int): The batch size for training.
            overwrite (bool): Whether to overwrite existing tuning results.
            model_file (Union[str, None], optional): The file path of the base model to load. Required if hyperModel is None. Defaults to None.
            learning_rate_max (Union[float, None], optional): The maximum learning rate for tuning. Required if hyperModel is None. Defaults to None.
            learning_rate_min (Union[float, None], optional): The minimum learning rate for tuning. Required if hyperModel is None. Defaults to None.
            training_optimizer (Union[str, None], optional): The optimizer used for training. Required if hyperModel is None. Defaults to None.
            training_loss (Union[str, None], optional): The loss function use for training. Required if hyperModel is None. Defaults to None.
            training_metrics (Union[list, None], optional): The metrics use for training. Required if hyperModel is None. Defaults to None.
            test_size (float, optional): The ratio of test data to split from the training data. Defaults to 0.2.
            max_trials (int, optional): The maximum number of trials for tuning. Defaults to 100.
            random_seed (int, optional): The random seed for reproducibility. Defaults to 42.
        Returns:
            tuple: A tuple containing the trained model and the evaluation score.
        """
        
        
        #Preparamos datos de entrenamiento
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

        #Si el hyperModel no está definido, lo definimos aqui
        if hyperModel is None:

            required_params = [model_file, learning_rate_max, learning_rate_min, training_optimizer, training_loss, training_metrics]
            if any(param is None for param in required_params):
                raise ValueError("You must define an hypermodel or the model_file, learning_rate_max, learning_rate_min, training_optimizer, training_loss, and training_metrics parameters")

            #Cargamos el modelo
            base_model = tf.keras.models.load_model(model_file, custom_objects=ak.CUSTOM_OBJECTS)

            #Creamos el hypermodel
            def hyperModel(hp):
                # Clonamos el modelo base para que en cada iteración se empiece desde el mismo punto
                model = tf.keras.models.clone_model(base_model)

                # Cargamos los pesos del modelo base
                model.set_weights(base_model.get_weights())

                # Definimos el número de capas a congelar
                num_layers_to_freeze = hp.Int('num_layers_to_freeze', min_value=0, max_value=len(model.layers) - 1)
                for layer in model.layers[:num_layers_to_freeze]:
                    layer.trainable = False

                # Definimos la tasa de aprendizaje
                learning_rate = hp.Float('learning_rate', min_value = learning_rate_min, max_value=learning_rate_max, step=10, sampling="log")

                optimizer = None
                if(training_optimizer == "adam"):
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                elif(training_optimizer == "sgd"):
                    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

                # Compilamos el modelo
                model.compile(
                    loss = training_loss,
                    optimizer = optimizer,
                    metrics = training_metrics
                )

                return model


        tuner = keras_tuner.GridSearch(
            hyperModel,
            objective=tunerObjectives,
            max_trials = max_trials,
            overwrite=overwrite,
            directory=tmp_dir,
            project_name = f'{modelName}_tuning',
            seed = random_seed
        )

        tuner.search(X_train, y_train, epochs=1000, validation_data= (X_val, y_val),
                     verbose=2,
                     batch_size=batch_size,
                     callbacks=callbacks)
        
        # Devolvemos el modelo entrenado
        model = tuner.get_best_models()[0]
        model.build(input_shape=(X.shape[1],))
        score = model.evaluate(X_val, y_val, verbose=0)

        # Imprimimos la mejor configuración
        best_hps = tuner.get_best_hyperparameters()[0]
        print(best_hps.values)

        return model, score