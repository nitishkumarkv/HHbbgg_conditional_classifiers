import os
import argparse
import subprocess
import yaml
import math
from pathlib import Path

from submission.condor_training import (
    apply_lightweight_test_preset,
    better_analyze_job,
    build_job_spec,
    ensure_job_files,
    submit_job,
    summarize_resource_matches,
)

def should_run_prepare_inputs(args):
    return any(
        [
            args.prep_inputs_for_training,
            args.prepare_inputs_pred_sim,
            args.prepare_inputs_pred_data,
            args.prepare_inputs_pred_sys,
        ]
    )


def should_run_training(args):
    return any(
        [
            args.perform_training,
            args.train_best_model,
            args.plot_training_results,
            args.get_permutation_importance,
            args.get_predictions,
            args.get_predictions_sys,
            args.test_mass_sculpting,
            args.get_data_mc_plots,
            args.get_score_shape_diff_kl,
        ]
    )


def should_run_categorisation(args):
    return args.perform_categorisation


def validate_requested_actions(args, parser):
    if not any(
        [
            should_run_prepare_inputs(args),
            should_run_training(args),
            should_run_categorisation(args),
            args.submit_training_to_condor,
            args.condor_better_analyze,
        ]
    ):
        parser.error(
            "No action was selected. Choose a pipeline step such as --prepare_inputs, "
            "--train_best_model, --perform_training, --perform_categorisation, or "
            "--condor_better_analyze."
        )

    if args.submit_training_to_condor and not (args.train_best_model or args.perform_training):
        parser.error(
            "--submit_training_to_condor only changes where the training step runs; "
            "it does not select a training step by itself. "
            "Add --train_best_model for a Condor training submission, or "
            "--perform_training to request the full training pipeline."
        )


def prepare_inputs(args, out_path_override=None, mhh_var=None, mhh_range=None):
    from data.prepare_inputs import PrepareInputs

    config_path = args.config_path
    out_path = out_path_override or args.out_path

    # create the output directory
    os.makedirs(out_path, exist_ok=True)

    input_vars_path = f"{config_path}/input_variables.yaml"

    # Load the configuration yaml files
    with open(f"{config_path}/training_config.yaml", 'r') as f:
        training_config = yaml.safe_load(f)

    prep_inputs = PrepareInputs(input_var_json=input_vars_path,
                                training_info = training_config,
                                outpath=out_path,
                                mhh_var=mhh_var,
                                mhh_range=mhh_range)
    
    # prepare the inputs for training
    if args.prep_inputs_for_training:
        print('INFO: Preparing the inputs for training', '\n')
        prep_inputs.prep_inputs_for_training()

    # prepare the inputs for prediction
    if args.prepare_inputs_pred_sim:
        print('INFO: Preparing the inputs for prediction', '\n')
        prep_inputs.prep_inputs_for_prediction_sim()

    # prepare the inputs for prediction data
    if args.prepare_inputs_pred_data:
        print('INFO: Preparing the inputs for prediction data', '\n')
        prep_inputs.prep_inputs_for_prediction_data()

    # prepare the inputs for systematics
    if args.prepare_inputs_pred_sys:
        print('INFO: Preparing the inputs for prediction systematics', '\n')
        prep_inputs.prep_inputs_for_prediction_sim_sys()

def perform_training(args):

    out_path = args.out_path
    config_path = args.config_path

    # Load the configuration yaml files
    training_config_path = f"{config_path}/training_config.yaml"
    with open(f"{training_config_path}", 'r') as f:
        training_config = yaml.safe_load(f)

    do_random_search = training_config["do_random_search"]

    if args.submit_training_to_condor and do_random_search:
        raise ValueError("Condor submission mode supports training only. Run random search locally first or disable do_random_search in the config.")

    # do random search
    if do_random_search:
        print('INFO: Performing random search')
        subprocess.run(f"python3 models/random_search.py --input_path {out_path} --training_config_path {training_config_path}", shell=True)        

    # perform trainging
    if args.train_best_model:
        if args.submit_training_to_condor:
            print('INFO: Submitting the best-model training to Condor')
            spec = build_job_spec(
                repo_root=Path(__file__).resolve().parent,
                out_path=out_path,
                input_path=out_path,
                training_config_path=training_config_path,
                tag=args.condor_tag,
                condor_work_dir=args.condor_work_dir,
                cpus=args.condor_cpus,
                memory_gb=args.condor_memory_gb,
                disk_gb=args.condor_disk_gb,
                gpus=args.condor_gpus,
                accounting_group=args.condor_accounting_group,
                job_flavor=args.condor_job_flavor,
                requirements=args.condor_requirements,
                n_epochs=args.n_epochs,
                schedd=args.condor_schedd,
                submission_mode=args.condor_submission_mode,
            )
            ensure_job_files(spec)
            print(f"INFO: Condor workspace: {spec.condor_root}")
            print(f"INFO: Condor submission mode: {spec.submission_mode}")
            print(f"INFO: Submit file: {spec.submit_path}")
            print(f"INFO: Wrapper script: {spec.wrapper_path}")
            print("INFO: Begin Condor submit file")
            print(spec.submit_path.read_text(encoding="utf-8"))
            print("INFO: End Condor submit file")
            print("INFO: Begin Condor wrapper script")
            print(spec.wrapper_path.read_text(encoding="utf-8"))
            print("INFO: End Condor wrapper script")
            maybe_print_condor_resource_diagnostics(args, spec)
            result = submit_job(spec, dry_run=args.condor_dry_run)
            if args.condor_dry_run:
                print('INFO: Dry-run enabled; submit files were rendered but no job was queued.')
            else:
                print(f"INFO: Submitted Condor cluster id: {result['cluster_id'] or 'unknown'}")
                if result.get("schedd"):
                    print(f"INFO: Submitted via schedd: {result['schedd']}")
                    print(f"INFO: Query with: condor_q -name {result['schedd']} -nobatch {result['cluster_id']}")
                elif result.get("cluster_id"):
                    print(f"INFO: Query with: condor_q -nobatch {result['cluster_id']}")
            return
        print('INFO: Training the best model')
        cmd = f"python3 models/training_utils.py --input_path {out_path} --training_config_path {training_config_path}"
        if args.n_epochs is not None:
            cmd += f" --n_epochs {args.n_epochs}"
        subprocess.run(cmd, shell=True)

    # plot the training results
    if args.plot_training_results:
        print('INFO: Getting the results plots')
        subprocess.run(f"python3 models/mlp_plotter.py --input_path {out_path}", shell=True)
        
    # get permutaion importance
    if args.get_permutation_importance:
        print('INFO: Getting permutation importance')
        subprocess.run(f"python3 models/permutation_importance.py --input_path {out_path}", shell=True)
    
    # get the predictions
    if args.get_predictions:
        print('INFO: Getting the predictions nominal')
        subprocess.run(f"python3 models/get_prediction.py --model_folder {out_path}/after_random_search_best1/ --samples_path {out_path} --config_path {config_path} --get_pred_nominal", shell=True)

    # get the predictions for systematics
    if args.get_predictions_sys:
        print('INFO: Getting the predictions systematics')
        subprocess.run(f"python3 models/get_prediction.py --model_folder {out_path}/after_random_search_best1/ --samples_path {out_path} --config_path {config_path} --get_pred_sys", shell=True)

    # get non resonant mass for different ggFHH score cuts
    if args.test_mass_sculpting:
        print('INFO: Getting non resonant mass for different ggFHH score cuts')
        subprocess.run(f"python3 utils/test_cor_mass.py --input_path {out_path} --config_path {config_path}", shell=True)

    # get the predictions for data
    if args.get_data_mc_plots:
        print('INFO: Getting data-MC plots')
        subprocess.run(f"python3 utils/plotting_utils.py --base-path {out_path} --training_config_path {training_config_path}", shell=True)

    # get score shapes for different kl samples
    if args.get_score_shape_diff_kl:
        print('INFO: Getting score shape differences')
        subprocess.run(f"python3 utils/score_shape_diff_kl.py --folder {out_path}/individual_samples/", shell=True)

def perform_categorisation(args):
    pass


def maybe_print_condor_resource_diagnostics(args, spec):
    if not args.condor_diagnose_resources:
        return

    print("INFO: Querying HTCondor pool for machines that satisfy this request")
    summary = summarize_resource_matches(spec)
    print(f"INFO: Match constraint: {summary['constraint']}")
    print(f"INFO: Available constraint: {summary['available_constraint']}")
    print(f"INFO: Capable slots: {summary['capable_slots']}")
    print(f"INFO: Capable machines: {summary['capable_machines']}")
    print(f"INFO: Currently available slots: {summary['available_slots']}")
    print(f"INFO: Currently available machines: {summary['available_machines']}")
    print(f"INFO: Example capable machines: {summary['capable_examples']}")
    print(f"INFO: Example available machines: {summary['available_examples']}")


def maybe_run_condor_better_analyze(args):
    if not args.condor_better_analyze:
        return False

    print(f"INFO: Running condor_q -better-analyze for cluster {args.condor_better_analyze}")
    if args.condor_schedd:
        print(f"INFO: Using schedd override {args.condor_schedd}")
    print(better_analyze_job(args.condor_better_analyze, schedd=args.condor_schedd))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform MLP based classification')
    parser.add_argument('--config_path', type=str, help='Path to the configuration files')
    parser.add_argument('--out_path', type=str, help='Path to save the inputs')
    parser.add_argument('--prep_inputs_for_training', action='store_true', help='Prepare inputs for training')
    parser.add_argument('--prepare_inputs_pred_sim', action='store_true', help='Prepare inputs for prediction')
    parser.add_argument('--prepare_inputs_pred_data', action='store_true', help='Prepare inputs for prediction data')
    parser.add_argument('--prepare_inputs_pred_sys', action='store_true', help='Prepare inputs for prediction systematics')
    parser.add_argument('--train_best_model', action='store_true', help='Train the best model')
    parser.add_argument('--plot_training_results', action='store_true', help='Plot training results')
    parser.add_argument('--get_permutation_importance', action='store_true', help='Get permutation importance')
    parser.add_argument('--get_predictions', action='store_true', help='Get predictions for nominal MC and data')
    parser.add_argument('--get_predictions_sys', action='store_true', help='Get predictions for systematics')
    parser.add_argument('--test_mass_sculpting', action='store_true', help='Test mass sculpting')
    parser.add_argument('--perform_training', action='store_true', help='Perform training')
    parser.add_argument('--get_data_mc_plots', action='store_true', help='Get data-MC plots')
    parser.add_argument('--perform_categorisation', action='store_true', help='Perform categorisation')
    parser.add_argument('--prepare_inputs', action='store_true', help='Prepare all inputs')
    parser.add_argument('--no_auto_prep_phase', action='store_true', help='Do not auto-run training-prep for all bins when downstream steps are requested')
    parser.add_argument('--mhh_bin', type=str, default=None, help='(Optional) Process only this mHH bin. Can be the bin index (0-based) or the bin name like "mHH_bin_0_to_350"')
    parser.add_argument('--get_score_shape_diff_kl', action='store_true', help='Get score shape differences using kl samples')
    parser.add_argument('--n_epochs', '--condor_epochs', dest='n_epochs', type=int, default=None, help='Optional epoch override for training, used in both local and Condor modes.')

    parser.add_argument('--submit_training_to_condor', action='store_true', help='Submit the training step to HTCondor instead of running it locally')
    parser.add_argument('--condor_work_dir', type=str, default=None, help='Optional directory for rendered Condor job files. Defaults to <out_path>/condor_runs/')
    parser.add_argument('--condor_tag', type=str, default=None, help='Optional tag to include in the Condor run directory name')
    parser.add_argument('--condor_cpus', type=int, default=4, help='Requested CPU cores for the Condor training job')
    parser.add_argument('--condor_memory_gb', type=int, default=32, help='Requested memory in GB for the Condor training job')
    parser.add_argument('--condor_disk_gb', type=int, default=20, help='Requested disk in GB for the Condor training job')
    parser.add_argument('--condor_gpus', type=int, default=1, help='Requested GPUs for the Condor training job')
    parser.add_argument('--condor_accounting_group', type=str, default=None, help='Optional HTCondor accounting group')
    parser.add_argument('--condor_job_flavor', dest='condor_job_flavor', type=str, default=None, help='Optional job flavor to include in the submit file.')
    parser.add_argument('--condor_requirements', type=str, default=None, help='Optional raw HTCondor requirements expression')
    parser.add_argument('--condor_schedd', type=str, default=None, help='Optional schedd override for condor_submit. If unset, use your normal HTCondor default routing.')
    parser.add_argument('--condor_submission_mode', type=str, choices=['spool', 'eossubmit'], default='spool', help='How to submit from lxplus/EOS: use standard schedds with condor_submit -spool (default) or load the CERN EosSubmit schedds.')
    parser.add_argument('--condor_lightweight_test', action='store_true', help='Submit a short real Condor training test: 1 epoch with the espresso job flavor (~20 minutes at CERN)')
    parser.add_argument('--condor_diagnose_resources', action='store_true', help='Query HTCondor to count machines/slots that can satisfy the requested CPU/GPU/memory/disk requirements')
    parser.add_argument('--condor_better_analyze', type=str, default=None, help='Run condor_q -better-analyze for the given cluster id and exit')
    parser.add_argument('--condor_dry_run', action='store_true', help='Render Condor job files without submitting a job')

    parser.add_argument('--do_all', action='store_true', help='Perform all steps')
    args = parser.parse_args()

    if maybe_run_condor_better_analyze(args):
        raise SystemExit(0)

    if args.do_all:
        args.prepare_inputs = True
        args.perform_training = True
        args.perform_categorisation = True

    if args.prepare_inputs:
        args.prep_inputs_for_training = True
        args.prepare_inputs_pred_sim = True
        args.prepare_inputs_pred_data = True
        args.prepare_inputs_pred_sys = False

    if args.condor_lightweight_test:
        apply_lightweight_test_preset(args)

    if args.perform_training:
        args.train_best_model = True
        args.plot_training_results = True
        args.get_permutation_importance = False
        args.get_predictions = True
        args.get_predictions_sys = False
        args.test_mass_sculpting = True
        args.get_data_mc_plots = True
        args.get_score_shape_diff_kl = True

    if args.submit_training_to_condor:
        args.plot_training_results = False
        args.get_permutation_importance = False
        args.get_predictions = False
        args.get_predictions_sys = False
        args.test_mass_sculpting = False
        args.get_data_mc_plots = False
        args.get_score_shape_diff_kl = False

    validate_requested_actions(args, parser)

    # load training config to check for mHH binning
    training_config_path = f"{args.config_path}/training_config.yaml"
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)

    mhh_binning = training_config.get('mHH_binning', None)

    if mhh_binning is None:
        # normal single-run behavior
        if should_run_prepare_inputs(args):
            prepare_inputs(args)
        if should_run_training(args):
            perform_training(args)
        if should_run_categorisation(args):
            perform_categorisation(args)
    else:
    # build bin edges: assume edges list defines internal edges, with implicit 0 and inf
        variable = mhh_binning.get('variable', None)
        edges = mhh_binning.get('edges', [])

        # validate edges
        if not isinstance(edges, list):
            raise ValueError('mHH_binning.edges must be a list of numbers')

        # construct boundaries [0, *edges, inf]
        boundaries = [0.0] + [float(x) for x in edges] + [math.inf]

        # Build a small list of bins (index, lo, hi, name) so we can select a specific bin if requested
        bins = []
        for i in range(len(boundaries)-1):
            lo = boundaries[i]
            hi = boundaries[i+1]
            hi_name = 'inf' if not math.isfinite(hi) else str(int(hi))
            bin_name = f"mHH_bin_{int(lo)}_to_{hi_name}"
            bins.append((i, lo, hi, bin_name))

        # If user specified --mhh_bin, resolve it to a single bin index list
        selected_bin_indices = [b[0] for b in bins]
        if args.mhh_bin is not None:
            key = args.mhh_bin
            found = None
            # try integer index
            try:
                idx = int(key)
                if 0 <= idx < len(bins):
                    found = idx
            except Exception:
                pass

            # try exact bin name
            if found is None:
                for (i, lo, hi, name) in bins:
                    if name == key or name.replace('mHH_bin_', '') == key:
                        found = i
                        break

            # try simple lo_hi form like "0_350" or "0-350"
            if found is None:
                key2 = key.replace('-', '_')
                for (i, lo, hi, name) in bins:
                    simple = f"{int(lo)}_{'inf' if not math.isfinite(hi) else int(hi)}"
                    if key2 == simple:
                        found = i
                        break

            if found is None:
                raise ValueError(f'Could not resolve --mhh_bin={args.mhh_bin} to any bin. Known bins: {[b[3] for b in bins]}')

            selected_bin_indices = [found]

        # Decide whether downstream work is requested (training/prediction/plots/etc.)
        needs_downstream = (
            args.prepare_inputs_pred_sim or args.prepare_inputs_pred_data or args.prepare_inputs_pred_sys
            or args.perform_training or args.train_best_model or args.plot_training_results
            or args.get_permutation_importance or args.get_predictions or args.get_predictions_sys
            or args.test_mass_sculpting or args.get_data_mc_plots or args.get_score_shape_diff_kl
        )

        # Phase 1: if downstream work is requested but the user did not explicitly ask for
        # training-prep, create training-prep artifacts for ALL bins first so later
        # prediction/training steps won't fail due to missing files. This can be disabled
        # with --no_auto_prep_phase.
        if needs_downstream and not args.prep_inputs_for_training and not args.no_auto_prep_phase:
            print("INFO: Detected downstream steps that require training input artifacts.")
            print("      Running training input preparation for all mHH bins (phase 1).")
            for (i, lo, hi, bin_name) in bins:
                # respect selected_bin_indices when user requested a specific bin
                if i not in selected_bin_indices:
                    continue

                per_bin_out = os.path.join(args.out_path, bin_name)

                print(f"INFO: Preparing training inputs for bin {bin_name}: {lo} <= {variable} < {hi}")

                # Build a temporary args object that only requests the training-prep step.
                tmp_args = argparse.Namespace(**vars(args))
                tmp_args.prep_inputs_for_training = True
                tmp_args.prepare_inputs_pred_sim = False
                tmp_args.prepare_inputs_pred_data = False
                tmp_args.prepare_inputs_pred_sys = False

                prepare_inputs(tmp_args, out_path_override=per_bin_out, mhh_var=variable, mhh_range=(lo, hi))

        # Phase 2: run the requested per-bin pipeline (may include prepare_inputs for
        # prediction, perform_training, and plotting). Default per-bin folder name is
        # `mHH_bin_{lo}_to_{hi}`.
        for (i, lo, hi, bin_name) in bins:
            # skip bins not requested when --mhh_bin was provided
            if i not in selected_bin_indices:
                continue

            per_bin_out = os.path.join(args.out_path, bin_name)

            print(f"INFO: Running pipeline for bin {bin_name}: {lo} <= {variable} < {hi}")

            # Prepare inputs for this bin (pass mhh var/range into PrepareInputs via prepare_inputs wrapper)
            if should_run_prepare_inputs(args):
                prepare_inputs(args, out_path_override=per_bin_out, mhh_var=variable, mhh_range=(lo, hi))

            # Run training and post-processing using this per-bin output directory
            # Temporarily override args.out_path for training steps
            old_out = args.out_path
            args.out_path = per_bin_out
            if should_run_training(args):
                perform_training(args)
            if should_run_categorisation(args):
                perform_categorisation(args)
            args.out_path = old_out
