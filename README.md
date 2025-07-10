

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
python3 run_multiclass_strategy.py --config_path config/VBF_version_20250524_MVAID/ --out_path VBF_version_20250524_MVAID/ --do_all
```
Config path has yaml files which define the input variables to use, the classes, the training config, etc.


# Example usage for categorisation

```
python3 categorisation/bayesian_categorization.py --n_categories 4 --base_path VBF_version_20250524_MVAID/ --n_runs {default=10}
```
`--n_run` defines the number of complete categorisation runs to choose the best one. The script also has several other useful arguments which you can have a look.
