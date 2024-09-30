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

import bertoferrero.neural_models.positioning.fingerprint.models as models

class ModelFactory:
    @staticmethod
    def create_model(model_type, inputlength, outputlength):
        if model_type == "M1":
            return models.M1(inputlength, outputlength)
        elif model_type == "M2":
            return models.M2(inputlength, outputlength)
        elif model_type == "M3":
            return models.M3(inputlength, outputlength)
        elif model_type == "M4":
            return models.M4(inputlength, outputlength)
        elif model_type == "M5":
            return models.M5(inputlength, outputlength)
        elif model_type == "M6":
            return models.M6(inputlength, outputlength)
        elif model_type == "M7":
            return models.M7(inputlength, outputlength)
        elif model_type == "M8":
            return models.M8(inputlength, outputlength)
        else:
            raise ValueError("Invalid model type")