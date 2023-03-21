# Government Intent

## Usage

#### Preprocess data

On `root` dir:

```sh
$ python preprocess.py [mode]
```

output processed data on `processed_data/` dir.

#### Train a model

On `root` dir:

```sh
$ python train.py --config_path config/[config_file]
```

The model will be saved under `output_dir` of config_file. Also, it will be zipped as `zipped_model_path` in config_file. 

#### Test a model

On `root` dir:

```sh
$ python test.py --config_path config/[config_file]
```

The result (`output.json`) will be saved under `output_dir` of config_file. 

#### Predict

See `nlu.py` for usage
