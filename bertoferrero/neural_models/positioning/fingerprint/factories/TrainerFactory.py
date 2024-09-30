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

import bertoferrero.neural_models.positioning.fingerprint.trainers as trainers

class TrainerFactory:
    @staticmethod
    def retrieve_trainer(model_type):
        if model_type == "M1":
            return trainers.M1Trainer
        elif model_type == "M2":
            return trainers.M2Trainer
        elif model_type == "M3":
            return trainers.M3Trainer
        elif model_type == "M4":   
            return trainers.M4Trainer
        elif model_type == "M5":
            return trainers.M5Trainer
        elif model_type == "M6":
            return trainers.M6Trainer
        elif model_type == "M7":
            return trainers.M7Trainer
        elif model_type == "M8":
            return trainers.M8Trainer
        else:
            raise ValueError("Invalid model type")