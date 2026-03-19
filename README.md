

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

Optional per-run environment activation settings live in `config/<runname>/environment.yaml`:

```yaml
manager: mamba
name: HHbbgg_classifier
```

If this file is absent, Condor submission defaults to `mamba` and `HHbbgg_classifier`.


# Condor training submission

Condor submission mode allows the execution of training jobs on condor clusters. Submission is assumed to be from EOS/lxplus, and the training job is assumed to be GPU-based. 

```
python3 run_multiclass_strategy.py \
  --config_path <config/Version...> \
  --out_path <Version...> \
  --perform_training \
  --submit_training_to_condor \
  --condor_accounting_group group_cms \
  --n_epochs 100 \
  --condor_diagnose_resources \
  --condor_job_flavor testmatch \
  --condor_requirements '!regexp(".*V100.*", GPUs_DeviceName)'
```

By default this uses standard lxplus schedd routing together with `condor_submit -spool`, so `condor_q` will follow the usual schedd assignment (e.g. `bigbird19.cern.ch`). If you do not pass `--condor_job_flavor`, submitted jobs default to `testmatch`.

To force submission to a specific schedd instead of your normal HTCondor default routing:

```
python3 run_multiclass_strategy.py \
  --config_path config/Version_20250524_MVAID_forPreApp/ \
  --out_path Version_20250524_MVAID_forPreApp/ \
  --train_best_model \
  --submit_training_to_condor \
  --condor_schedd bigbird24.cern.ch \
  --condor_requirements '!regexp(".*V100.*", GPUs_DeviceName)'
```

To use the CERN EosSubmit schedds instead of the default spool submission mode:

```
python3 run_multiclass_strategy.py \
  --config_path config/Version_20250524_MVAID_forPreApp/ \
  --out_path Version_20250524_MVAID_forPreApp/ \
  --train_best_model \
  --submit_training_to_condor \
  --condor_submission_mode eossubmit \
  --condor_requirements '!regexp(".*V100.*", GPUs_DeviceName)'
```

Using `eossubmit` may route the job to a different schedd than your normal default, like `bigbird24.cern.ch`, so be aware that it will not show up in `condor_q` on your normal default schedd. Run `condor_q -submitter <your_username> -schedd bigbird24.cern.ch` to check the status of your job if you use `eossubmit`, for example. 

Submit a lightweight real Condor test job that trains for one epoch and asks HTCondor to schedule it as `espresso` (20 minutes runtime):

```
python3 run_multiclass_strategy.py \
  --config_path config/Version_20250524_MVAID_forPreApp/ \
  --out_path Version_20250524_MVAID_forPreApp/ \
  --condor_lightweight_test \
  --condor_accounting_group group_cms \
  --condor_requirements '!regexp(".*V100.*", GPUs_DeviceName)'
```

For a quick smoke test, limit the job to one epoch first:

```
python3 run_multiclass_strategy.py \
  --config_path config/Version_20250524_MVAID_forPreApp/ \
  --out_path Version_20250524_MVAID_forPreApp/ \
  --train_best_model \
  --submit_training_to_condor \
  --n_epochs 1 \
  --condor_job_flavor espresso \
  --condor_requirements '!regexp(".*V100.*", GPUs_DeviceName)' \
  --condor_dry_run
```

The submission files, wrapper, metadata, and logs are written under `<out_path>/condor_runs/<timestamp>_<tag>_<id>/`. The wrapper bootstraps the CMS environment with `scram`, activates the python virtual environment (`config/<runname>/environment.yaml`), and runs `models/training_utils.py`. In the default `spool` mode, HTCondor keeps the job log/stdout/stderr on the schedd until you fetch them with `condor_transfer_data`; in `eossubmit` mode they are written back to EOS automatically.


# Condor dataset preparation

Dataset preparation can be run locally (default) or on condor.

Local run for all prep steps:

```
python3 run_multiclass_strategy.py \
  --config_path <config/Version...> \
  --out_path <Version...> \
  --prepare_inputs
```

Submit all prepare steps to condor:

```
python3 run_multiclass_strategy.py \
  --config_path <config/Version...> \
  --out_path <Version...> \
  --prepare_inputs \
  --submit_prepare_to_condor \
  --condor_accounting_group group_cms \
  --condor_job_flavor tomorrow \
  --condor_cpus 4 \
  --condor_memory_gb 32 \
  --condor_disk_gb 20 \
  --condor_diagnose_resources
```

The argument `--condor_diagnose_resources` prints out the resource requests that will be used for the prepare jobs, which is helpful to check before submission. Manually setting the resource requests is also possible, as shown above.

For single prepare subcommands, run exactly the one you want:

```
python3 run_multiclass_strategy.py \
  --config_path <config/Version...> \
  --out_path <Version...> \
  --prep_inputs_for_training \
  --submit_prepare_to_condor \
  --condor_accounting_group group_cms
```

```
python3 run_multiclass_strategy.py \
  --config_path <config/Version...> \
  --out_path <Version...> \
  --prepare_inputs_pred_sim \
  --submit_prepare_to_condor \
  --condor_accounting_group group_cms
```

```
python3 run_multiclass_strategy.py \
  --config_path <config/Version...> \
  --out_path <Version...> \
  --prepare_inputs_pred_data \
  --submit_prepare_to_condor \
  --condor_accounting_group group_cms
```

```
python3 run_multiclass_strategy.py \
  --config_path <config/Version...> \
  --out_path <Version...> \
  --prepare_inputs_pred_sys \
  --submit_prepare_to_condor \
  --condor_accounting_group group_cms
```

If you need to debug something, fast turnaround time is important. `--condor_lightweight_prepare_test` runs the prepare code on a minimal dataset pulled from the training config. To run a fast lightweight prepare test on condor, use:

```
python3 run_multiclass_strategy.py \
  --config_path <config/Version...> \
  --out_path Version_20250524_MVAID_forPreApp \
  --prepare_inputs \
  --submit_prepare_to_condor \
  --condor_lightweight_prepare_test \
  --condor_accounting_group group_cms
```

`--condor_lightweight_prepare_test` uses:
- `--condor_prepare_max_input_files 1` by default
- `--condor_prepare_max_rows_per_file 20000` by default
- `--condor_job_flavor espresso`
- `--condor_gpus 0`

You can override those values manually:

```
--condor_prepare_max_input_files 2
--condor_prepare_max_rows_per_file 5000
```

For direct testing of a single prepare mode only, you can call `run_prepare_inputs.py` from submit jobs:

```
python3 run_prepare_inputs.py \
  --config_path <config/Version...> \
  --out_path <Version...> \
  --prep_inputs_for_training \
  --max_input_files 1 \
  --max_rows_per_file 20000
```

To ask HTCondor why a queued job is idle:

```
python3 run_multiclass_strategy.py --condor_better_analyze 124674 --condor_schedd bigbird24.cern.ch
```


# Example usage for categorisation

```
python3 categorisation/bayesian_categorization.py --n_categories 4 --base_path Version_20250524_MVAID_forPreApp/ --n_runs {default=15}
```
`--n_run` defines the number of complete categorisation runs to choose the best one. The script also has several other useful arguments which you can have a look.
