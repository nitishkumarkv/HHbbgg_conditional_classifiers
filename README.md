

## Installing the micromamba environment

```
micromamba env create -f env.yml
```

Execute this to activate the environment:
```
micromamba activate HHbbgg_classifier
```


# Example usage

```
python3 run_multiclass_strategy.py --config_path config/Version_20250524_MVAID_forPreApp/ --out_path Version_20250524_MVAID_forPreApp/ --do_all
```
Config path has yaml files which define the input variables to use, the classes, the training config, etc.


# Example usage for categorisation

```
python3 categorisation/bayesian_categorization.py --n_categories 4 --base_path Version_20250524_MVAID_forPreApp/ --n_runs {default=15}
```
`--n_run` defines the number of complete categorisation runs to choose the best one. The script also has several other useful arguments which you can have a look.

# Adding columns to prediction
- Change `self.extra_vars_out` in L47 of data/prepare_inputs.py
- If you wish to change the columns for specific eras or samples, see training_config.yaml in the config folder, commenting out the eras or samples which you don't need.
- Rerun the prediction with the following. Note that the out_path argument takes the folder which was used for training. BEcause of this, the parquets will be overwritten.
```
python3 run_multiclass_strategy.py --config_path config/CONFIG_FOLDER/ --out_path OUT_PATH_WITH_TRAINING  --prepare_inputs_pred_sim --prepare_inputs_pred_data --get_predictions
```
- Add columns to mergerScript.py, in `columns` (L41) if the variable is in data, and in `columns_gen` (L99) if the variable is not in data.
- Adjust `dict_run_eras` (L283) if you are running on particular eras.
- Rerun mergerScript with `python mergerScript.py OUT_PATH_WITH_TRAINING'
