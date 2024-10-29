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
        
    def build_model(self, empty_values: bool = False):
        raise NotImplementedError

    def build_model_autokeras(self, designing:bool, overwrite:bool, tuner:str , random_seed:int, autokeras_project_name:str, auokeras_folder:str, max_trials:int = 100):
        raise NotImplementedError
        