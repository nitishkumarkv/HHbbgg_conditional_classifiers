

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
python3 MLP_multiclass.py --config_path config/MLP_multiclass/ --samples_path /eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet/v2/Run3_2022 --out_path test_MLP
```
