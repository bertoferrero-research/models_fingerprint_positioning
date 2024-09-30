
# Neural Network Models for Fingerprinting

This library contains a collection of eight neural network models (M1-M8) and their corresponding training classes (M1Trainer - M8Trainer) used for positioning fingerprinting. These models have been designed to be compatible with AutoML libraries. This code serves as a demonstration of the models utilized in our research paper, which is currently under review.

## Usage

### Training a Model

Below is an example of how to train a neural network model using the provided classes. This example assumes that you have a dataset prepared and accessible in the specified paths.

```python
import json
import os
import tempfile
from bertoferrero.neural_models.positioning.fingerprint.factories import TrainerFactory
from bertoferrero.neural_models.positioning.fingerprint.trainingcommon import save_model

# General configuration
test_name = "Test4"
fingerprint_name = "20240805_090525_145639_rssi"
scaler_filename = "scaler.pkl"
model_filename = "model.keras"
score_filename = "score.json"
tuner = "bayesian"
batch_size = 256
max_trials = 50
random_seed = 42

# Define general paths
script_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.join(script_dir, test_name)
fingerprint_path = os.path.join(working_dir, "dataset", "fingerprint", fingerprint_name)
models_dir = os.path.join(working_dir, "models")

models = ['M2']  # List of models to train

# Iterate over each model
for model in models:
    model_working_dir = os.path.join(models_dir, model)
    os.makedirs(model_working_dir, exist_ok=True)

    variant_name = 'with_empty_values'  # Example variant
    print("---- Training " + model + " " + variant_name + " ----")

    variant_working_dir = os.path.join(model_working_dir, variant_name)
    os.makedirs(variant_working_dir, exist_ok=True)

    model_path = os.path.join(variant_working_dir, model_filename)
    scaler_path = os.path.join(variant_working_dir, scaler_filename)
    dataset_path = os.path.join(fingerprint_path, variant_name + ".csv")
    score_path = os.path.join(variant_working_dir, score_filename)
    model_tmp_dir = tempfile.mkdtemp()

    # Load the trainer class
    trainerClass = TrainerFactory.retrieve_trainer(model)

    # Train the model
    model_obj, score = trainerClass.train_model(
        dataset_path=dataset_path,
        scaler_file=scaler_path,
        tuner=tuner,
        tmp_dir=model_tmp_dir,
        batch_size=batch_size,
        designing=False,
        overwrite=True,
        max_trials=max_trials,
        random_seed=random_seed
    )

    # Save the model
    save_model(model_obj, model_path)

    # Save the score
    with open(score_path, 'w') as f:
        json.dump(score, f, indent=4)

print("Training completed successfully!")
```

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/new-feature`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to your branch (`git push origin feature/new-feature`).
5. Open a Pull Request on this repository.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.

## Related Work

The models presented in this library are part of a research paper that is currently under review. Once published, a link to the paper will be provided here for further reference.
