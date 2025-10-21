# Multiclassifier for HH->bbgg analysis

## Description

Dataset and file names follow these conventions:
`w` - event weights
`X` - input features for classifier (e.g. kinematic and ID variables)
`y` - target classes (e.g. signal and various background classes)
`z` - variable to decorrelate from (e.g. invariant mass of the system)


## Installing the micromamba environment

```bash
micromamba env create -f env.yml
```

Execute this to activate the environment:
```bash
micromamba activate HHbbgg_classifier
```


# Example usage

```bash
python3 run_multiclass_strategy.py --config_path config/Version_20250524_MVAID_forPreApp/ --out_path Version_20250524_MVAID_forPreApp/ --do_all
```
Config path has yaml files which define the input variables to use, the classes, the training config, etc.


# Example usage for categorisation

```bash
python3 categorisation/bayesian_categorization.py --n_categories 4 --base_path Version_20250524_MVAID_forPreApp/ --n_runs {default=15}
```
`--n_run` defines the number of complete categorisation runs to choose the best one. The script also has several other useful arguments which you can have a look.


# Sculpting Study

## Example Workflow

```bash
# Re-run prep inputs to build X feature set
python3 run_multiclass_strategy.py --config_path config/Version_20250524_MVAID_forPreApp/ --out_path Version_20250524_MVAID_forPreApp/ --prep_inputs_for_training

# Prepare sculpting study inputs (y target set)
python3 run_multiclass_strategy.py --config_path config/Version_20250524_MVAID_forPreApp/ --out_path Version_20250524_MVAID_forPreApp/ --prepare_sculpting_study_inputs

# Train Mjj predictor DNN
python3 run_multiclass_strategy.py --config_path config/Version_20250524_MVAID_forPreApp/ --out_path Version_20250524_MVAID_forPreApp/ --train_mjj_predictor

# Get permutation importance for Mjj predictor
python3 run_multiclass_strategy.py --config_path config/Version_20250524_MVAID_forPreApp/ --out_path Version_20250524_MVAID_forPreApp/ --mjj_predictor_permutation_importance
```