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
python3 run_multiclass_strategy.py --config_path config/Version_20250524_MVAID_forPreApp/ --out_path Version_20250524_MVAID_forPreApp/ --perform_categorisation
```
Categorisation is configured by the `categorization_config.yaml` file in the config path. 
- `n_run` defines the number of complete categorisation runs to choose the best one.
- `base_path` defines where to find input samples (e.g. `Version_20250524_MVAID_forPreApp/`)
- `optuna_folder` name of output folder for optuna results (e.g. `optuna_categorization`)
- `n_categories` number of categories to create
- `n_trials` number of optuna trials to run for each categorisation run
- `SR_strategy` categorization strategy for signal region (options: `sequential`, `simultaneous`)
- `gamma_strategy` gamma strategy for TPE sampler (options: `sqrt`, `linear`)


# Sculpting Study

## Example Workflow

1. Prepare inputs for sculpting study (X feature set)
2. Prepare inputs for sculpting study (y target set, i.e. Mjj values)
3. Train sculpting DNN to predict Mjj
4. Get permutation importance for Mjj predictor

```bash
python3 run_multiclass_strategy.py --config_path config/Version_20250524_MVAID_forPreApp/ --out_path Version_20250524_MVAID_forPreApp/ --prep_inputs_for_training --prepare_sculpting_study_inputs --train_mjj_predictor --mjj_predictor_permutation_importance
```

# Wishlist
## Batch Jobs / Condor
- Optuna categorization: Add option for N-runs submits N jobs in parallel to condor, pick best one after all finish (final condor job separate, waits for all N to finish)
- Submit multiclass model for training on condor
    - Submit multiple jobs for hyperparameter scan, pick best model and save hyperparams to json

## Checkpoints
- Add option to save y_pred_train.npy, y_train.npy, y_pred_val.npy, and y_val.npy to each checkpoint directory for ROC plots
- Add option to mlp_plotter to load from checkpoint dirs for ROC plots
- Add option to run multiclass permutation importance at each checkpoint (also in condor job)
- Add option to plot permutation importance evolution over training epochs (each feature is a line on the plot and they evolve over epochs, like loss/accuracy plots)

## Additional Features
- SHAP feature importance (multiclass and sculpting study)
- K-fold cross validation