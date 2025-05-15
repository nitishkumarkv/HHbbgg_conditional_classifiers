

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
python3 MLP_multiclass.py --config_path config/example_config/ --do_all
```

Config path has json files which define the input variables to use, the classes... 