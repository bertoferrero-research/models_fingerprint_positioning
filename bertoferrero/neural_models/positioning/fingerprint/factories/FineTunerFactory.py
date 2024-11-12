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

import bertoferrero.neural_models.positioning.fingerprint.finetuners as tuners

class FineTunerFactory:
    @staticmethod
    def retrieve_finetuner(model_type):
        """
        Retrieve the appropriate fine-tuner class based on the given model type.

        Args:
            model_type (str): The type of the model. Expected values are "M2", "M4", or "M7".

        Returns:
            class: The fine-tuner class corresponding to the specified model type.

        Raises:
            ValueError: If the provided model type is not one of the expected values.
        """
        if model_type == "M2":
            return tuners.M2FineTuner
        elif model_type == "M4":   
            return tuners.M4FineTuner
        elif model_type == "M7":
            return tuners.M7FineTuner
        else:
            raise ValueError("Invalid model type")