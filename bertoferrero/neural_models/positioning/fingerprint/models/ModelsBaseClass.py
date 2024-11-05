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
import autokeras as ak
import tensorflow as tf

class ModelsBaseClass:

    def __init__(self, inputlength, outputlength):
        self.inputlength = inputlength
        self.outputlength = outputlength

    @staticmethod
    def load_traning_data(data_file: str, scaler_file: str):
        raise NotImplementedError

    @staticmethod
    def load_testing_data(data_file: str, scaler_file: str):
        raise NotImplementedError
        
    def build_model(self, random_seed:int, empty_values: bool = False, base_model_path: str = None):
        raise NotImplementedError

    def build_model_autokeras(self, designing:bool, overwrite:bool, tuner:str , random_seed:int, autokeras_project_name:str, auokeras_folder:str, max_trials:int = 100):
        raise NotImplementedError
    
    def _build_model_from_base(self, base_model_path: str, loss, metrics):
        #Cargamos el modelo
        model_original = tf.keras.models.load_model(base_model_path, custom_objects=ak.CUSTOM_OBJECTS)

        # Obtener el optimizador del modelo
        optimizer = model_original.optimizer

        # Copia limpia
        json_model = model_original.to_json()
        model = tf.keras.models.model_from_json(json_model)

        # Compilamos
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        return model
        