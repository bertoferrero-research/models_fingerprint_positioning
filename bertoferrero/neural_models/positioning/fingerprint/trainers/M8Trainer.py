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

from bertoferrero.neural_models.positioning.fingerprint.models import M8
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from .BaseTrainer import BaseTrainer
import tensorflow as tf
import numpy as np
from bertoferrero.neural_models.positioning.fingerprint.trainingcommon import posXYlist_to_grid, gridList_to_posXY
import keras_tuner
from bertoferrero.neural_models.positioning.fingerprint.trainingcommon import descale_numpy

class M8Trainer(BaseTrainer):
    @staticmethod
    def train_model(dataset_path: str, scaler_file: str, tuner: str, tmp_dir: str, batch_size: int, designing: bool, overwrite: bool, max_trials:int = 100, random_seed: int = 42, hyperparams_log_path: str = None):
               
        #Definimos el nombre del modelo y la configuración específica
        modelName = 'M8'
        cell_amount_x = 3
        cell_amount_y = 3
        max_trials = 100

        #Cargamos los datos de entrenamiento
        X, y = M8.load_traning_data(dataset_path, scaler_file)

        #Cargamos cada dimension
        y_dim1 = posXYlist_to_grid(y.to_numpy(), cell_amount_x, cell_amount_y)
        y_dim2 = posXYlist_to_grid(y.to_numpy(), cell_amount_x**2, cell_amount_y**2)

        #Convertimos a categorical
        y_dim1 = tf.keras.utils.to_categorical(y_dim1, num_classes=cell_amount_x*cell_amount_y)
        y_dim2 = tf.keras.utils.to_categorical(y_dim2, num_classes=(cell_amount_x**2)*(cell_amount_y**2))
        
        #Instanciamos la clase del modelo
        inputlength = X.shape[1]
        outputlength_dim1 = y_dim1.shape[1]
        outputlength_dim2 = y_dim2.shape[1]
        modelInstance = M8(inputlength, [outputlength_dim1, outputlength_dim2])

        #Creamos la función hypermodelo para el entrenamiento
        def build_model(hp):
            #Definimos los hipermodelos y construimos el modelo en si
            loss = 'categorical_crossentropy'
            activation = 'softmax'
            inputdropout = hp.Float('input_dropout', min_value=0.0, max_value=0.5, step=0.1)
            dimension1 = []
            dimension2 = []
            if designing:
                for i in range(hp.Int('num_layers_dimension1', min_value=1, max_value=10, step=1)):
                    dimension1.append(
                        {
                            'units': hp.Int('units_dimension1_'+str(i), min_value=4, max_value=2048, step=2, sampling="log"), 
                            'dropout': hp.Float('dropout_dimension1_'+str(i), min_value=0.0, max_value=0.5, step=0.1)
                        })

                for i in range(hp.Int('num_layers_dimension2', min_value=1, max_value=10, step=1)):
                    dimension2.append(
                        {
                            'units': hp.Int('units_dimension2_'+str(i), min_value=4, max_value=2048, step=2, sampling="log"),
                            'dropout': hp.Float('dropout_dimension2_'+str(i), min_value=0.0, max_value=0.5, step=0.1)
                        }
                    )
                loss_weight_d1 = hp.Float('loss_weight_d1', min_value=0.1, max_value=0.9, step=0.1)
                learning_rate = learning_rate=hp.Float('learning_rate', min_value=0.00001, max_value=0.1, step=10, sampling="log")

            else:
                d1units = [512, 512, 512, 64, 256, 2048, 128, 256]
                for i in range(len(d1units)):
                    dimension1.append(
                        {
                            'units': d1units[i], 
                            'dropout': hp.Float('dropout_dimension1_'+str(i), min_value=0.0, max_value=0.5, step=0.1)
                        })
                
                d2units = [32, 128, 128, 64]
                for i in range(len(d2units)):
                    dimension2.append(
                        {
                            'units': d2units[i], 
                            'dropout': hp.Float('dropout_dimension2_'+str(i), min_value=0.0, max_value=0.5, step=0.1)
                        })
                loss_weight_d1 = 0.2
                learning_rate = 0.0001            

            loss_weight_d2 = 1 - loss_weight_d1
            
            return modelInstance.build_custom_model(dimension1=dimension1, dimension2=dimension2, loss_weight_d1=loss_weight_d1, loss_weight_d2=loss_weight_d2, inputdropout=inputdropout, learning_rate=learning_rate, loss=loss, activation_d1=activation, activation_d2=activation)
            

        #Entrenamos
        callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
        callback2 = tf.keras.callbacks.EarlyStopping(monitor='val_output_d2_accuracy', min_delta=0.0001, patience=10, restore_best_weights=True)
        callback3 = tf.keras.callbacks.EarlyStopping(monitor='val_output_d1_accuracy', min_delta=0.0001, patience=10, restore_best_weights=True)
        callbackTensor = tf.keras.callbacks.TensorBoard(tmp_dir+"/"+modelName+"/tensorboard_logs")
        X_train, X_val, y_dim1_train, y_dim1_val, y_dim2_train, y_dim2_val = train_test_split(X, y_dim1, y_dim2, test_size=0.2)
        keras_tuner.GridSearch
        tuner = keras_tuner.BayesianOptimization(
            build_model,
            objective=[keras_tuner.Objective("val_output_d2_accuracy", direction="max"), keras_tuner.Objective("val_output_d1_accuracy", direction="max"), keras_tuner.Objective("val_loss", direction="min")],
            max_trials=max_trials,
            overwrite=overwrite,
            directory=tmp_dir,
            project_name=modelName,
            seed=random_seed
        )

        tuner.search(X_train, [y_dim1_train, y_dim2_train], epochs=1000, validation_data=(X_val, [y_dim1_val, y_dim2_val]), 
                        verbose=2,
                        batch_size=batch_size,
                        callbacks=[callback1, callback2, callback3, callbackTensor])

        #Registramos hiperparámetros
        if(hyperparams_log_path is not None):
            BaseTrainer.automl_trials_logger(tuner, hyperparams_log_path, max_trials)

        # Devolvemos el modelo entrenado
        model = tuner.get_best_models()[0]
        model.build(input_shape=(inputlength,))
        score = model.evaluate(X_val, [y_dim1_val, y_dim2_val], verbose=0)

        # Imprimimos la mejor configuración
        best_hps = tuner.get_best_hyperparameters()[0]
        print(best_hps.values)

        return model, score

    @staticmethod
    def prediction(dataset_path: str, model_file: str, scaler_file: str):
        cell_amount_x = 3
        cell_amount_y = 3

        #Cargamos los datos de entrenamiento
        input_data, output_data = M8.load_testing_data(dataset_path, scaler_file)
        output_data = output_data.to_numpy()

        #Cargamos el modelo
        model = tf.keras.models.load_model(model_file)

        #Predecimos
        predictions = model.predict(input_data)
        #Nos quedamos con la salida final
        predictions = predictions[-1]
        #Usamp argmax para quedarnos con la probabilidad más alta de cada posición
        predictions = np.argmax(predictions, axis=-1)
        # Convertimos a posiciones
        predictions_positions = gridList_to_posXY(predictions, cell_amount_x=cell_amount_x**2, cell_amount_y=cell_amount_y**2)

        # Evaluación
        output_data_grid_dim1 = posXYlist_to_grid(output_data, cell_amount_x, cell_amount_y)
        output_data_grid_dim2 = posXYlist_to_grid(output_data, cell_amount_x**2, cell_amount_y**2)
        output_data_categorical_dim1 = tf.keras.utils.to_categorical(output_data_grid_dim1, num_classes=cell_amount_x*cell_amount_y)
        output_data_categorical_dim2 = tf.keras.utils.to_categorical(output_data_grid_dim2, num_classes=(cell_amount_x**2)*(cell_amount_y**2))

        #accuracy = accuracy_score(output_data_grid_dim2, predictions)
        metrics = model.evaluate(input_data, [output_data_categorical_dim1, output_data_categorical_dim2], verbose=0)
        formated_metrics = {
            'loss_mse': metrics[0],
            'loss_d1': metrics[1],
            'loss_d2': metrics[2],
            'accuracy_d1': metrics[3],
            'accuracy_d2': metrics[4]
        }

        #Devolvemos las predicciones y los datos de salida esperados
        return predictions_positions, output_data, formated_metrics