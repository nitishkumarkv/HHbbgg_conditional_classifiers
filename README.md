

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
python3 run_multiclass_strategy.py --config_path config/example_config/ --out_path example_folder --do_all
```

Config path has yaml files which define the input variables to use, the classes, the training config, etc.