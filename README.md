

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
  --config_path config/Version_20250524_MVAID_forPreApp/ \
  --out_path Version_20250524_MVAID_forPreApp/ \
  --train_best_model \
  --submit_training_to_condor \
  --condor_accounting_group group_cms \
  --condor_job_flavor tomorrow
```

By default this uses standard lxplus schedd routing together with `condor_submit -spool`, so `condor_q` will follow the usual schedd assignment (e.g. `bigbird19.cern.ch`).

To force submission to a specific schedd instead of your normal HTCondor default routing:

```
python3 run_multiclass_strategy.py \
  --config_path config/Version_20250524_MVAID_forPreApp/ \
  --out_path Version_20250524_MVAID_forPreApp/ \
  --train_best_model \
  --submit_training_to_condor \
  --condor_schedd bigbird24.cern.ch
```

To use the CERN EosSubmit schedds instead of the default spool submission mode:

```
python3 run_multiclass_strategy.py \
  --config_path config/Version_20250524_MVAID_forPreApp/ \
  --out_path Version_20250524_MVAID_forPreApp/ \
  --train_best_model \
  --submit_training_to_condor \
  --condor_submission_mode eossubmit
```

Using `eossubmit` may route the job to a different schedd than your normal default, like `bigbird24.cern.ch`, so be aware that it will not show up in `condor_q` on your normal default schedd. Run `condor_q -submitter <your_username> -schedd bigbird24.cern.ch` to check the status of your job if you use `eossubmit`, for example. 

Submit a lightweight real Condor test job that trains for one epoch and asks HTCondor to schedule it as `espresso` (20 minutes runtime):

```
python3 run_multiclass_strategy.py \
  --config_path config/Version_20250524_MVAID_forPreApp/ \
  --out_path Version_20250524_MVAID_forPreApp/ \
  --condor_lightweight_test \
  --condor_accounting_group group_cms
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
  --condor_dry_run
```

The submission files, wrapper, metadata, and logs are written under `<out_path>/condor_runs/<timestamp>_<tag>_<id>/`. The wrapper bootstraps the CMS environment with `scram`, activates the python virtual environment (`config/<runname>/environment.yaml`), and runs `models/training_utils.py`. In the default `spool` mode, HTCondor keeps the job log/stdout/stderr on the schedd until you fetch them with `condor_transfer_data`; in `eossubmit` mode they are written back to EOS automatically.

To ask HTCondor why a queued job is idle:

```
python3 run_multiclass_strategy.py --condor_better_analyze 124674 --condor_schedd bigbird24.cern.ch
```


# Example usage for categorisation

```
python3 categorisation/bayesian_categorization.py --n_categories 4 --base_path Version_20250524_MVAID_forPreApp/ --n_runs {default=15}
```
`--n_run` defines the number of complete categorisation runs to choose the best one. The script also has several other useful arguments which you can have a look.
