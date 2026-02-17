import copy
import os
import sys
import json
import yaml
import pickle
import re
import warnings
import argparse
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import awkward as ak
import mplhep

try:
    from models.mlp import MLP
except ImportError:
    try:
        from mlp import MLP
    except ImportError:
        raise ImportError("Could not import MLP class from models")

plt.style.use(mplhep.style.CMS)

def load_checkpoint(file_path):
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_loss_hist = checkpoint['train_loss_hist']
    train_loss_hist_no_absolute = checkpoint['train_loss_hist_no_absolute_weights']

    val_loss_hist = checkpoint['val_loss_hist']
    train_acc_hist = checkpoint['train_acc_hist']
    val_acc_hist = checkpoint['val_acc_hist']
    lr_hist = checkpoint['lr_hist']
    best_weights = checkpoint['best_weights']
    best_loss = checkpoint['best_loss']
    start_epoch = checkpoint['epoch']
    print(f'Checkpoint loaded from {file_path}, resuming from epoch {start_epoch + 1}')
    return checkpoint
    # return start_epoch, train_loss_hist, train_loss_hist_no_absolute, val_loss_hist, train_acc_hist, val_acc_hist, lr_hist, best_weights, best_loss




def _in_venv() -> bool:
    """
    Checks if the current Python interpreter is running inside a virtual environment.

    Returns:
        bool: True if in a virtual environment, False otherwise.
    """
    base_prefix = getattr(sys, 'base_prefix', None)
    real_prefix = getattr(sys, 'real_prefix', None)
    prefix = sys.prefix

    if real_prefix is not None:
        # legacy virtualenv
        return True
    if base_prefix is not None and prefix != base_prefix:
        return True
    if os.environ.get("VIRTUAL_ENV", ""):
        return True
    
    return False


def _get_python_env_base(job_config: dict) -> str:
    """
    Gets the base path of the current Python environment (conda or venv).

    Args:
        job_config (dict): Job configuration dictionary from yaml file.

    Returns:
        str: Base path of the Python environment.
    
    Raises:
        EnvironmentError: If not running inside a virtual environment.
    """
    if not _in_venv() and job_config.get('conda_env', None) is None:
        raise EnvironmentError(
            "Current Python interpreter is not running inside a virtual environment and no conda environment specified in job config. "
            + "Please activate the appropriate conda/mamba environment before submitting this condor job or add appropriate configuration."
        )

    if job_config.get('conda_env', None) is not None:
        config_conda_env = job_config['conda_env']
        if not os.path.exists(config_conda_env):
            raise FileNotFoundError(f"Conda environment path specified in job_config not found: {config_conda_env}")
        return config_conda_env

    # Check if conda
    conda_prefix = os.environ.get('CONDA_PREFIX', None)
    if conda_prefix is not None:
        return conda_prefix
    
    # Otherwise, return venv
    return sys.prefix


def write_sub_file(
        condor_dir: str, 
        plot_dir: str, 
        checkpoint_path: str, 
        input_path: str, 
        script_path: str, 
        model_folder: str, 
        **kwargs
) -> str:
    """
    Writes a condor submission file with the given output path and keyword arguments.

    Args:
        condor_dir (str): Directory to store condor scripts and submission files (relative path). 
        plot_dir (str): Directory to save plots (relative path). 
        checkpoint_path (str): Path to the model checkpoint, including filename (relative path).
        input_path (str): Path to the input data (relative path).
        script_path (str): Path to the script to be executed (relative path).
        model_folder (str): Folder name inside input_path where the model and plots are stored (nominally "after_random_search_best1").
    
    Kwargs:
        epoch (int, optional): Specifies epoch for output file naming/labeling, not which input data is selected.
        verbose (bool, optional): If True, enables verbose condor submission output.

    Returns:
        str: Path to submission file.
    """
    SUBMITTED_ON_EOS = os.getcwd().startswith('/eos/')
    CWD = os.getcwd()
    
    sub_file_name = 'mlp_plotter.sub'
    sub_file_name_no_ext = os.path.splitext(sub_file_name)[0]
    sub_file_path = os.path.join(condor_dir, sub_file_name)
    os.makedirs(os.path.dirname(sub_file_path), exist_ok=True)

    # For condor, we need to escape the arguments differently
    # Condor expects arguments to be space-separated, with quotes escaped
    plotter_args = f'--input_path {input_path} --checkpoint_file {checkpoint_path} --path_for_plots {plot_dir} --model_folder {model_folder}'
    out_dest = os.path.join(CWD, condor_dir)
    if SUBMITTED_ON_EOS:
        out_dest = f"root://eosuser.cern.ch/{out_dest}"

    extra  = ""
    if kwargs.get('epoch', None) is not None:
        extra += f"_epoch{kwargs['epoch']}"

    # script_path = os.path.join(condor_dir, 'mlp_plotter.sh')

    with open(sub_file_path, 'w') as f:
        f.write(f"# {sub_file_name}\n")
        f.write("\n")
        f.write("universe = vanilla\n")
        f.write(f"executable    = {script_path}\n")
        f.write(f"arguments     = {plotter_args}\n")
        f.write(f"output        = out/{sub_file_name_no_ext}_job_$(Cluster)_$(Process){extra}.out\n")
        f.write(f"error         = err/{sub_file_name_no_ext}_job_$(Cluster)_$(Process){extra}.err\n")
        f.write(f"log           = log/{sub_file_name_no_ext}_job_$(Cluster)_$(Process){extra}.log\n")
        f.write("\n")
        f.write("RequestCPUs     = 1\n")
        f.write("RequestGPUs     = 1\n")
        f.write("+RequiresGPUs   = 1\n")
        f.write("RequestMemory   = 4GB\n")
        f.write("RequestDisk     = 2GB\n")
        f.write('Requirements    = regexp("^NVIDIA .100.*$", GPUs_DeviceName)\n')  # Only for the filtered test
        f.write("\n")
        f.write("getenv = True\n")
        f.write(f"initialdir = {CWD}\n")
        f.write("should_transfer_files = NO\n")
        # f.write("should_transfer_files = YES\n")
        # f.write("transfer_input_files = models/mlp_plotter.py, models/mlp.py, models/__init__.py\n")
        # f.write("when_to_transfer_output = ON_EXIT\n")
        f.write(f"output_destination = {out_dest}\n") # EOS requires -spool -> requires "output_destination" or "transfer_output_files" (transfer plugin)
        f.write("+JobFlavor = \"espresso\"\n")
        f.write("\n")
        f.write("queue\n")

    if kwargs.get('verbose', False):
        print("[DEBUG] Condor submission file written to:", sub_file_path)
    return sub_file_path


def write_condor_script(condor_dir: str, job_config: dict) -> str:
    """
    Writes a condor script with the given script path and keyword arguments.

    Args:
        condor_dir (str): Directory to store condor scripts and submission files.
        job_config (dict): Job configuration dictionary from yaml file.

    Returns:
        str: Path to condor script.
    """
    os.makedirs(condor_dir, exist_ok=True)

    env_base = _get_python_env_base(job_config)
    cwd = os.getcwd()

    script_path = os.path.join(condor_dir, 'mlp_plotter.sh')

    # plotter_args = f"--input_path {os.path.dirname(checkpoint_path)} --checkpoint_path {checkpoint_path} --path_for_plots {plot_dir}"

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")  # Exit on any error
        f.write("set -x\n")  # Print commands as they execute
        f.write("echo \"=== JOB STARTING ===\"\n")
        f.write("echo \"Host: $(hostname)\"\n")
        f.write("echo \"User: $(whoami)\"\n")
        f.write("echo \"Date: $(date)\"\n")
        f.write("echo \"PWD: $(pwd)\"\n")
        f.write("echo \"Arguments: $@\"\n")
        f.write("\n")
        f.write(f"cd {cwd} || exit 1\n")
        f.write("\n")
        f.write("# Test Python executable\n")
        f.write("PYTHON_EXE='python3'\n")
        f.write("if command -v python3 &> /dev/null; then\n")
        f.write("    PYTHON_EXE='python3'\n")
        f.write("elif command -v python &> /dev/null; then\n")
        f.write("    PYTHON_EXE='python'\n")
        f.write("else\n")
        f.write("    echo \"ERROR: No Python executable found\"\n")
        f.write("    exit 1\n")
        f.write("fi\n")
        f.write("\n")
        f.write("echo \"Using Python: $PYTHON_EXE\"\n")
        f.write("$PYTHON_EXE --version\n")
        f.write("\n")
        f.write("# Test if script exists\n")
        f.write("if [ -f models/mlp_plotter.py ]; then\n")
        f.write("    SCRIPT_PATH='models/mlp_plotter.py'\n")
        f.write("elif [ -f mlp_plotter.py ]; then\n")
        f.write("    SCRIPT_PATH='mlp_plotter.py'\n")
        f.write("else\n")
        f.write("    echo \"ERROR: mlp_plotter.py not found\"\n")
        f.write("    ls -la || echo \"Current directory listing failed\"\n")
        f.write("    exit 1\n")
        f.write("fi\n")
        f.write("\n")
        f.write("echo \"Using script: $SCRIPT_PATH\"\n")
        f.write("\n")
        f.write("echo \"=== RUNNING PYTHON SCRIPT ===\"\n")
        f.write("$PYTHON_EXE \"$SCRIPT_PATH\" \"$@\"\n")
        f.write("EXIT_CODE=$?\n")
        f.write("echo \"=== JOB FINISHED WITH EXIT CODE: $EXIT_CODE ===\"\n")
        f.write("\n")
        f.write("exit $EXIT_CODE\n")

    # Make the script executable
    os.chmod(script_path, 0o755)
    
    return script_path
        


def submit_condor_job(sub_path: str, clargs: argparse.Namespace | None) -> None:
    """
    Submits a condor job with the given script and keyword arguments.

    Args:
        sub_path (str): Path to the condor submission file.
        args (argparse.Namespace | None, optional): Command-line arguments. Condor verbosity control.
    """
    # Construct the command with keyword arguments
    if clargs is not None and clargs.verbose: # lets args be an optional arg
        cmd = f"condor_submit -verbose -spool {sub_path}"
    else:
        cmd = f"condor_submit -spool {sub_path}"

    print("[INFO] Submitting condor job with command:", cmd)
    result = os.system(cmd)
    
    if result == 0:
        print("[INFO] Job submitted successfully!")
    else:
        print(f"[ERROR] Job submission failed with exit code: {result}")
        print(f"[INFO] Check the submission file: {sub_path}")
        # Print the submission file contents for debugging
        print("[DEBUG] Submission file contents:")
        try:
            with open(sub_path, 'r') as f:
                for i, line in enumerate(f, 1):
                    print(f"{i:2d}: {line.rstrip()}")
        except Exception as e:
            print(f"[ERROR] Could not read submission file: {e}")


def run_condor_job(
        input_path: str = "",                       # required
        condor_dir: str = "",                       # required
        plot_dir: str = "",                         # required
        checkpoint_file: str = "",                  # required
        job_config: dict = {},                      # required
        model_folder: str = "after_random_search_best1",  # optional
        clargs: argparse.Namespace | None = None,   # optional
        **kwargs
    ) -> None:
    """
    Runs a condor job for plotting MLP results.

    Steps:
        1. Write the condor submission file.
        2. Write the condor bash script.
        3. Submit condor job.

    Args:
        input_path (str): Base directory for the DNN. Usually inside of the project directory. (e.g. .../HHbbgg_conditional_classifiers/Version_20250524_MVAID_forPreApp/)
        condor_dir (str): Directory to store condor scripts and submission files.
        plot_dir (str): Directory to save plots.
        checkpoint_file (str): Path to the model checkpoint, including filename.
        job_config (dict): Job configuration dictionary from yaml file.
        clargs (argparse.Namespace | None, optional): Command-line arguments. Condor verbosity control.
    
    Kwargs:
        dry_run (bool, optional): If True, only writes condor files but does not submit the job.
        epoch (int, optional): Specific epoch of plot. Controls output file naming/labeling, not which input data is selected.
    """
    config_conda_env: str | None = job_config.get('conda_env', None)
    verbose: bool = clargs.get('verbose', False) if clargs is not None else False

    # Validate
    if not input_path:
        raise ValueError("Input directory 'input_path' must be provided. (e.g. .../HHbbgg_conditional_classifiers/Version_20250524_MVAID_forPreApp/)")
    if not condor_dir:
        raise ValueError("Condor directory 'condor_dir' must be provided. This is where condor scripts and submission files will be stored.")
    if not plot_dir:
        raise ValueError("Plot directory 'plot_dir' must be provided. This is where the output plots will be saved.")
    if not checkpoint_file:
        raise ValueError("Checkpoint file 'checkpoint_file' must be provided. This is the path to the model checkpoint, including filename.")
    if not job_config or not isinstance(job_config, dict):
        raise ValueError("Job configuration 'job_config' must be provided as a dictionary.")

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")


    if config_conda_env is not None:
        if not os.path.exists(config_conda_env):
            raise FileNotFoundError(f"Conda environment path specified in job_config not found: {config_conda_env}")
    elif not _in_venv() and config_conda_env is None:
        raise EnvironmentError(
            "Current Python interpreter is not running inside a virtual environment. "
            + "Please activate the appropriate conda/mamba environment before submitting this condor job."
        )

    script_path = write_condor_script(condor_dir, job_config)
    sub_path = write_sub_file(condor_dir, plot_dir, checkpoint_file, input_path, script_path, model_folder, epoch=kwargs.get('epoch', None), verbose=verbose)
    if kwargs.get('dry_run', False):
        print(f"mlp_plotter condor job dry run. Condor submission file written to: {sub_path}, script written to: {script_path}")
    else:
        submit_condor_job(sub_path, clargs)


if __name__ == "__main__":
    import argparse
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description='Plot the results of the MLP')
    parser.add_argument('--input_path', type=str, help='Path to the inputs')
    
    # Optional arguments
    parser.add_argument('--checkpoint_file', default=None, type=str, help='Path to the model checkpoint. If not provided, will use default path in <input_path>/after_random_search_best1/mlp.pth')
    parser.add_argument('--path_for_plots', default=None, type=str, help='Path to save the plots. If not provided, will use default path in <input_path>/after_random_search_best1/plots/')
    parser.add_argument('--model_folder', default='after_random_search_best1', type=str, help='Folder name inside input_path where the model and plots are stored. Default: after_random_search_best1')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output for debugging')

    # Batch-only arguments
    parser.add_argument('--batch', type=str, default='local', help='Batch submission system to use ("local" for no batch)')
    parser.add_argument('--job_config', default=None, type=str, help='Path to job configuration YAML file (for batch submission). Default: config/<input_path>/job_config.yaml')
    parser.add_argument('--dry_run', action='store_true', help='If set, will only write condor files but not submit the job (only relevant if --batch is set)')
    
    args = parser.parse_args()

    inputs_for_MLP = args.input_path
    input_path = f"{inputs_for_MLP}/{args.model_folder}/"

    if args.path_for_plots is None:
        path_for_plots = f"{input_path}/plots/"
    else:
        path_for_plots = args.path_for_plots
    
    os.makedirs(path_for_plots, exist_ok=True)
        
    if args.checkpoint_file is None:
        path_to_checkpoint = f"{input_path}/mlp.pth"
    else:
        path_to_checkpoint = args.checkpoint_file 
        if not path_to_checkpoint.endswith('.pth'):
            warnings.warn(
                f"Filename '{path_to_checkpoint}' does not have the expected '.pth' extension. "
                + "Please verify that this is the correct checkpoint file.",
                UserWarning
            )
    
    assert os.path.exists(path_to_checkpoint), f"Checkpoint file not found: {path_to_checkpoint}"

    # Extract epoch number from plot path
    epoch_num = None
    match = re.search(r'/epoch(\d+)/', path_for_plots) # Match '/epochX/' where X is the epoch number
    if match:
        epoch_num = int(match.group(1))
        print(f"[INFO] Extracted epoch {epoch_num} from plot path.")

    # Condor batch submission via command line
    if args.batch == 'condor':
        if args.job_config is None: # use default path
            job_config_path = f"config/{args.input_path}/job_config.yaml"
        else:
            job_config_path = args.job_config

        assert os.path.exists(job_config_path), f"Job config file not found: {job_config_path}"
        with open(f"{job_config_path}", 'r') as f:
            job_config = yaml.safe_load(f)
        run_condor_job(
            input_path=inputs_for_MLP,
            condor_dir=f"{input_path}/condor/mlp_plotter/",
            plot_dir=path_for_plots,
            checkpoint_file=path_to_checkpoint,
            job_config=job_config,
            dry_run=args.dry_run,
            epoch=epoch_num,
            model_folder=args.model_folder,
            clargs=args,
        )
        print(f"Saving plots to: {path_for_plots}")
        sys.exit(0)
    
    # load the checkpoint
    # start_epoch, train_loss_hist, train_loss_hist_no_absolute_weights, val_loss_hist, train_acc_hist, val_acc_hist, lr_hist, best_weights, best_loss = load_checkpoint(path_to_checkpoint)
    print("Loading checkpoint...")
    checkpoint = load_checkpoint(path_to_checkpoint)

    train_loss_hist:                                    list[float]         = checkpoint['train_loss_hist']
    train_loss_hist_no_absolute_weights:                list[float]         = checkpoint['train_loss_hist_no_absolute_weights']
    train_loss_hist_no_dist_corr:                       list[float] | None  = checkpoint.get('train_loss_hist_no_dist_corr', None)
    train_loss_hist_no_absolute_weights_no_dist_corr:   list[float] | None  = checkpoint.get('train_loss_hist_no_absolute_weights_no_dist_corr', None)
    train_dist_corr_hist:                               list[float] | None  = checkpoint.get('train_dist_corr_hist', None)

    val_loss_hist:              list[float]         = checkpoint['val_loss_hist']
    val_loss_hist_no_dist_corr: list[float] | None  = checkpoint.get('val_loss_hist_no_dist_corr', None)
    val_dist_corr_hist:         list[float] | None  = checkpoint.get('val_dist_corr_hist', None)

    best_weights:   dict[str, torch.Tensor] = checkpoint['best_weights']
    best_loss:      float                   = checkpoint['best_loss']
    best_dist_corr: float | None            = checkpoint.get('best_dist_corr', None)
    start_epoch:    int                     = checkpoint['epoch']

    train_acc_hist: list[float] = checkpoint['train_acc_hist']
    val_acc_hist:   list[float] = checkpoint['val_acc_hist']
    lr_hist:        list[float] = checkpoint['lr_hist']
    disco_in_loss:  bool | None = checkpoint.get('disco_in_loss', None)

    colors = ['royalblue', 'darkorange', 'darkviolet', 'seagreen']

    #plot loss function
    plt.plot(train_loss_hist, label="train")
    plt.plot(val_loss_hist, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.savefig(f'{path_for_plots}/loss_plot.png')
    plt.clf()

    plt.plot(train_loss_hist_no_absolute_weights, label="train")
    plt.plot(val_loss_hist, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.savefig(f'{path_for_plots}/loss_plot_no_abs.png')
    plt.clf()

    # if distance correlation was used, plot those too
    if disco_in_loss:
        assert train_loss_hist_no_dist_corr is not None, "If distance correlation was used in loss, train_loss_hist_no_dist_corr must be in checkpoint."
        assert train_loss_hist_no_absolute_weights_no_dist_corr is not None, "If distance correlation was used in loss, train_loss_hist_no_absolute_weights_no_dist_corr must be in checkpoint."
        assert train_dist_corr_hist is not None, "If distance correlation was used in loss, train_dist_corr_hist must be in checkpoint."
        assert val_loss_hist_no_dist_corr is not None, "If distance correlation was used in loss, val_loss_hist_no_dist_corr must be in checkpoint."
        assert val_dist_corr_hist is not None, "If distance correlation was used in loss, val_dist_corr_hist must be in checkpoint."
        assert best_dist_corr is not None, "If distance correlation was used in loss, best_dist_corr must be in checkpoint."

        plt.plot(train_loss_hist_no_dist_corr, label="train (no dist corr)")
        plt.plot(val_loss_hist_no_dist_corr, label="validation (no dist corr)")
        plt.xlabel("epochs")
        plt.ylabel("cross entropy")
        plt.legend()
        plt.savefig(f'{path_for_plots}/loss_plot_no_disco.png')
        plt.clf()
        
        plt.plot(train_loss_hist_no_dist_corr, label="train (excluding dist corr)")
        plt.plot(val_loss_hist_no_dist_corr, label="validation (excluding dist corr)")
        plt.plot(train_loss_hist, label="train (including dist corr)")
        plt.plot(val_loss_hist, label="validation (including dist corr)")
        plt.xlabel("epochs")
        plt.ylabel("cross entropy")
        plt.legend()
        plt.savefig(f'{path_for_plots}/loss_plot_DisCo_comparison.png')
        plt.clf()

        plt.plot(train_dist_corr_hist, label="train dist corr")
        plt.plot(val_dist_corr_hist, label="validation dist corr")
        plt.xlabel("epochs")
        plt.ylabel("distance correlation")
        plt.legend()
        plt.savefig(f'{path_for_plots}/dist_corr_plot.png')
        plt.clf()

        # Plot comparing loss with dist corr, loss without dist corr, and dist corr
        # Loss axes on left, dist corr axes on right
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(train_loss_hist, 'b-', label="train loss (incl dist corr)")
        ax1.plot(val_loss_hist, 'b--', label="val loss (incl dist corr)")
        ax1.plot(train_loss_hist_no_dist_corr, 'g-', label="train loss (excl dist corr)")
        ax1.plot(val_loss_hist_no_dist_corr, 'g--', label="val loss (excl dist corr)")
        ax2.plot(train_dist_corr_hist, 'r-', label="train dist corr")
        ax2.plot(val_dist_corr_hist, 'r--', label="val dist corr")
        ax1.set_xlabel("epochs")
        ax1.set_ylabel("cross entropy", color='b')
        ax2.set_ylabel("distance correlation", color='r')
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
        fig.tight_layout()
        plt.savefig(f'{path_for_plots}/loss_and_dist_corr_plot.png')
        plt.clf()
    
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(train_loss_hist_no_absolute_weights, 'b-', label="train loss (incl dist corr)")
        ax1.plot(train_loss_hist_no_absolute_weights_no_dist_corr, 'g-', label="train loss (excl dist corr)")
        ax2.plot(train_dist_corr_hist, 'r-', label="train dist corr")
        ax2.plot(val_dist_corr_hist, 'r--', label="val dist corr")
        ax1.set_xlabel("epochs")
        ax1.set_ylabel("cross entropy", color='b')
        ax2.set_ylabel("distance correlation", color='r')
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
        fig.tight_layout()
        plt.savefig(f'{path_for_plots}/loss_no_absolute_weights_and_dist_corr_plot.png')
        plt.clf()


    # plot learning rate
    plt.plot(lr_hist)
    plt.xlabel("epochs")
    plt.ylabel("learning rate")
    plt.savefig(f'{path_for_plots}/lr_plot.png')
    plt.clf()

    #plot accuracy
    plt.plot(train_acc_hist, label="train")
    plt.plot(val_acc_hist, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(f'{path_for_plots}/acc_plot.png')
    plt.clf()

    # TODO: Switch pathing for loading predictions from checkpoint dirs

    # load predictions
    y_pred_val_ = np.load(f"{input_path}/y_pred_val.npy")
    y_val_ = np.load(f'{inputs_for_MLP}/y_val.npy')
    #rel_w_val = np.load(f'{inputs_for_MLP}/rel_w_val.npy')
    rel_w_val_ = np.load(f'{inputs_for_MLP}/class_weights_for_val.npy')
    y_pred_val = y_pred_val_
    y_val = y_val_
    rel_w_val = rel_w_val_


    y_pred_train = np.load(f"{input_path}/y_pred_train.npy")
    y_train = np.load(f'{inputs_for_MLP}/y_train.npy')
    rel_w_train = np.load(f'{inputs_for_MLP}/true_class_weights.npy')

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from itertools import combinations

    # Class names
    class_names = ["non_resonant_bkg", "ttH", "other_single_H", "GluGluToHH", "VBFToHH_sig"]
    #n_classes = len(class_names)
    n_classes = y_val.shape[1]

    # One-vs-All ROC Curves
    one_vs_all_auc_dict = {}
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        class_name = class_names[i]
        y_true_binary = y_val[:, i]
        y_score = y_pred_val[:, i]
        # Compute ROC curve and ROC area
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=rel_w_val)
        fpr, tpr = zip(*sorted(zip(fpr, tpr)))
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:0.4f})')

        # Store the AUC for this class
        one_vs_all_auc_dict[f"{class_name}_fpr"] = fpr
        one_vs_all_auc_dict[f"{class_name}_tpr"] = tpr

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{path_for_plots}/roc_curve_one_vs_all.png')

    plt.xlim([0.0001, 1.0])
    plt.xscale('log')
    plt.savefig(f'{path_for_plots}/roc_curve_one_vs_all_logx.png')
    plt.clf()

    # save auc scores
    with open(f'{path_for_plots}/roc_curve_one_vs_all.json', 'w') as f:
        json.dump(one_vs_all_auc_dict, f)


    # One-vs-One ROC Curves
    # For each pair of classes
    glu_idx = class_names.index("GluGluToHH")

    # List of other class indices
    other_classes = [i for i in range(n_classes) if i != glu_idx]

    # Iterate over GluGluToHH vs each other class individually
    plt.figure(figsize=(8, 6))

    GluGluToHH_one_vs_one_roc = {}
    # Iterate over GluGluToHH vs each other class individually
    for j in other_classes:
        i = glu_idx  # Index of GluGluToHH
        class_name_i = class_names[i]
        class_name_j = class_names[j]
        # Select samples belonging to class i or class j
        idx = (y_val[:, i] == 1) | (y_val[:, j] == 1)
        y_true_binary = y_val[idx, i]
        y_score = y_pred_val[idx, i]  # Use the probability for class i (GluGluToHH)
        weights = rel_w_val[idx]
        # Compute ROC curve and ROC area
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=weights)
        fpr, tpr = zip(*sorted(zip(fpr, tpr)))
        roc_auc = auc(fpr, tpr)
        # Plot the ROC curve on the same figure
        plt.plot(fpr, tpr, label=f'{class_name_i} vs {class_name_j} (AUC = {roc_auc:0.4f})')

        # store the AUC
        GluGluToHH_one_vs_one_roc[f"{class_name}_fpr"] = fpr
        GluGluToHH_one_vs_one_roc[f"{class_name}_tpr"] = tpr

    # Plot the diagonal line representing random guessing
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    #plt.title(f'ROC Curves: {class_name_i} vs Each Other Class Individually', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f'{path_for_plots}/roc_curve_{class_name_i}_vs_all_individual.png')

    plt.xlim([0.0001, 1.0])
    plt.xscale('log')
    plt.savefig(f'{path_for_plots}/roc_curve_{class_name_i}_vs_all_individual_logx.png')
    plt.clf()

    #save the auc scores
    with open(f'{path_for_plots}/GluGluToHH_vs_all.json', 'w') as f:
        json.dump(GluGluToHH_one_vs_one_roc, f)

    if n_classes>4:
        # One-vs-One ROC Curves
        # For each pair of classes
        glu_idx = class_names.index("VBFToHH_sig")

        VBFToHH_one_vs_one_roc = {}
        # List of other class indices
        other_classes = [i for i in range(n_classes) if i != glu_idx]

        # Iterate over GluGluToHH vs each other class individually
        plt.figure(figsize=(8, 6))

        # Iterate over GluGluToHH vs each other class individually
        for j in other_classes:
            i = glu_idx  # Index of GluGluToHH
            class_name_i = class_names[i]
            class_name_j = class_names[j]
            # Select samples belonging to class i or class j
            idx = (y_val[:, i] == 1) | (y_val[:, j] == 1)
            y_true_binary = y_val[idx, i]
            y_score = y_pred_val[idx, i]  # Use the probability for class i (GluGluToHH)
            weights = rel_w_val[idx]
            # Compute ROC curve and ROC area
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=weights)
            fpr, tpr = zip(*sorted(zip(fpr, tpr)))
            roc_auc = auc(fpr, tpr)
            # Plot the ROC curve on the same figure
            plt.plot(fpr, tpr, label=f'{class_name_i} vs {class_name_j} (AUC = {roc_auc:0.4f})')

            # store the AUC
            VBFToHH_one_vs_one_roc[f"{class_name}_fpr"] = fpr
            VBFToHH_one_vs_one_roc[f"{class_name}_tpr"] = tpr

        # Plot the diagonal line representing random guessing
        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR', fontsize=12)
        plt.ylabel('TPR', fontsize=12)
        #plt.title(f'ROC Curves: {class_name_i} vs Each Other Class Individually', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True)
        plt.tight_layout()

        # Save the combined plot
        plt.savefig(f'{path_for_plots}/roc_curve_{class_name_i}_vs_all_individual.png')

        plt.xlim([0.0001, 1.0])
        plt.xscale('log')
        plt.savefig(f'{path_for_plots}/roc_curve_{class_name_i}_vs_all_individual_logx.png')

        plt.clf()

        # save AUC scores
        with open(f'{path_for_plots}/VBFToHH_vs_all.json', 'w') as f:
            json.dump(VBFToHH_one_vs_one_roc, f)

    y_val = y_train
    y_pred_val = y_pred_train
    rel_w_val = rel_w_train

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        class_name = class_names[i]
        y_true_binary = y_val[:, i]
        y_score = y_pred_val[:, i]
        # Compute ROC curve and ROC area
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=rel_w_val)
        fpr, tpr = zip(*sorted(zip(fpr, tpr)))
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:0.4f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    #plt.title('One-vs-All ROC Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{path_for_plots}/train_roc_curve_one_vs_all.png')
    plt.xlim([0.0001, 1.0])
    plt.xscale('log')
    plt.savefig(f'{path_for_plots}/train_roc_curve_one_vs_all_logx.png')
    plt.clf()

    if n_classes>4:
            
        # One-vs-One ROC Curves
        # For each pair of classes
        glu_idx = class_names.index("VBFToHH_sig")
    
        # List of other class indices
        other_classes = [i for i in range(n_classes) if i != glu_idx]
    
        # Iterate over GluGluToHH vs each other class individually
        plt.figure(figsize=(8, 6))
    
        # Iterate over GluGluToHH vs each other class individually
        for j in other_classes:
            i = glu_idx  # Index of GluGluToHH
            class_name_i = class_names[i]
            class_name_j = class_names[j]
            # Select samples belonging to class i or class j
            idx = (y_val[:, i] == 1) | (y_val[:, j] == 1)
            y_true_binary = y_val[idx, i]
            y_score = y_pred_val[idx, i]  # Use the probability for class i (GluGluToHH)
            weights = rel_w_val[idx]
            # Compute ROC curve and ROC area
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=weights)
            fpr, tpr = zip(*sorted(zip(fpr, tpr)))
            roc_auc = auc(fpr, tpr)
            # Plot the ROC curve on the same figure
            plt.plot(fpr, tpr, label=f'{class_name_i} vs {class_name_j} (AUC = {roc_auc:0.4f})')
    
        # Plot the diagonal line representing random guessing
        plt.plot([0, 1], [0, 1], 'k--')
    
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR', fontsize=12)
        plt.ylabel('TPR', fontsize=12)
        #plt.title(f'ROC Curves: {class_name_i} vs Each Other Class Individually', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True)
        plt.tight_layout()
    
        # Save the combined plot
        plt.savefig(f'{path_for_plots}/train_roc_curve_{class_name_i}_vs_all_individual.png')
        plt.close()

    # One-vs-One ROC Curves
    # For each pair of classes
    glu_idx = class_names.index("GluGluToHH")

    # List of other class indices
    other_classes = [i for i in range(n_classes) if i != glu_idx]

    # Iterate over GluGluToHH vs each other class individually
    plt.figure(figsize=(8, 6))

    # Iterate over GluGluToHH vs each other class individually
    for j in other_classes:
        i = glu_idx  # Index of GluGluToHH
        class_name_i = class_names[i]
        class_name_j = class_names[j]
        # Select samples belonging to class i or class j
        idx = (y_val[:, i] == 1) | (y_val[:, j] == 1)
        y_true_binary = y_val[idx, i]
        y_score = y_pred_val[idx, i]  # Use the probability for class i (GluGluToHH)
        weights = rel_w_val[idx]
        # Compute ROC curve and ROC area
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score, sample_weight=weights)
        fpr, tpr = zip(*sorted(zip(fpr, tpr)))
        roc_auc = auc(fpr, tpr)
        # Plot the ROC curve on the same figure
        plt.plot(fpr, tpr, label=f'{class_name_i} vs {class_name_j} (AUC = {roc_auc:0.4f})')

    # Plot the diagonal line representing random guessing
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    #plt.title(f'ROC Curves: {class_name_i} vs Each Other Class Individually', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f'{path_for_plots}/train_roc_curve_{class_name_i}_vs_all_individual.png')
    plt.close()


    colours = ['blue', 'red', 'green', 'orange', 'purple']


    plt.style.use(mplhep.style.CMS)  # Use CMS-like style

    colours = ['blue', 'red', 'green', 'orange', 'purple']


    colours = ['blue', 'red', 'green', 'orange', 'purple']

    for i in range(n_classes):
        fig, ax = plt.subplots(figsize=(8, 6))
        class_name = class_names[i]

        max_y = 0  # Track max y for ylim

        # --- TRAIN: step plot with shaded uncertainty ---
        for j in range(n_classes):
            mask = y_train[:, j] == 1
            y_vals = y_pred_train[mask, i]
            weights = rel_w_train[mask]
            weights_sq = weights**2

            hist_raw, bin_edges = np.histogram(y_vals, bins=25, weights=weights, range=(0, 1))
            hist_sq_raw, _ = np.histogram(y_vals, bins=bin_edges, weights=weights_sq, range=(0, 1))
            bin_widths = np.diff(bin_edges)

            total_weight = np.sum(hist_raw)
            if total_weight == 0:
                continue  # Avoid division by zero for empty bins/classes

            # Normalize to density
            hist_density = hist_raw / (total_weight * bin_widths)
            uncertainty_density = np.sqrt(hist_sq_raw) / (total_weight * bin_widths)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            max_y = max(max_y, np.max(hist_density + uncertainty_density))

            # Step line
            ax.step(
                bin_centers,
                hist_density,
                where='mid',
                label=f'Train {class_names[j]}',
                color=colours[j],
                linewidth=2,
            )

            # Shaded uncertainty band
            ax.fill_between(
                bin_centers,
                hist_density - uncertainty_density,
                hist_density + uncertainty_density,
                step='mid',
                color=colours[j],
                alpha=0.3,
            )

        # --- VALIDATION: dots with error bars ---
        for j in range(n_classes):
            mask = y_val_[:, j] == 1
            y_vals = y_pred_val_[mask, i]
            weights = rel_w_val_[mask]
            weights_sq = weights**2

            hist_raw, bin_edges = np.histogram(y_vals, bins=25, weights=weights, range=(0, 1))
            hist_sq_raw, _ = np.histogram(y_vals, bins=bin_edges, weights=weights_sq, range=(0, 1))
            bin_widths = np.diff(bin_edges)

            total_weight = np.sum(hist_raw)
            if total_weight == 0:
                continue

            hist_density = hist_raw / (total_weight * bin_widths)
            uncertainty_density = np.sqrt(hist_sq_raw) / (total_weight * bin_widths)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            max_y = max(max_y, np.max(hist_density + uncertainty_density))

            ax.errorbar(
                bin_centers,
                hist_density,
                yerr=uncertainty_density,
                fmt='o',
                label=f'Valid {class_names[j]}',
                color=colours[j],
                markersize=5,
                capsize=2,
                elinewidth=1,
            )

        # Labels and style
        ax.set_xlabel(f'{class_name} score')
        ax.set_ylabel('a.u.')
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-3, top=max_y * 100)
        ax.set_xlim(left=0, right=1)

        ax.legend(ncol=2, fontsize=10)
        # Uncomment this if you want the CMS label
        # mplhep.cms.label(loc=0, data=True, label='Preliminary')

        fig.tight_layout()
        fig.savefig(f'{path_for_plots}/{class_name}_score.png')
        plt.close(fig)
        
