import getpass
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from submission.condor_training import (
    apply_lightweight_test_preset,
    better_analyze_job,
    build_job_spec,
    build_submit_command,
    build_machine_constraint,
    load_environment_config,
    parse_submit_schedd_from_log,
    render_submit_file,
    render_wrapper_script,
    validate_submit_target,
)

TEST_USER = getpass.getuser()
TEST_USER_INITIAL = TEST_USER[0].lower()


def infer_eos_project_root() -> str:
    cwd = Path.cwd().resolve()
    cwd_str = str(cwd)
    if cwd_str.startswith("/eos/user/"):
        return cwd_str

    home_prefix = f"/eos/home-{TEST_USER_INITIAL}/{TEST_USER}/"
    if cwd_str.startswith(home_prefix):
        suffix = cwd_str[len(home_prefix):]
        return f"/eos/user/{TEST_USER_INITIAL}/{TEST_USER}/{suffix}"

    warnings.warn(
        "Current working directory is not on EOS. "
        "Synthesizing EOS-style test paths from the current project directory name.",
        stacklevel=2,
    )
    return f"/eos/user/{TEST_USER_INITIAL}/{TEST_USER}/{cwd.name}"


EOS_PROJECT_ROOT = infer_eos_project_root()
EOS_REPO_ROOT = f"{EOS_PROJECT_ROOT}/repo"
EOS_OUT_PATH = f"{EOS_PROJECT_ROOT}/out"
EOS_INPUT_PATH = EOS_OUT_PATH
EOS_TRAINING_CONFIG_PATH = f"{EOS_PROJECT_ROOT}/cfg/training_config.yaml"
EOS_OUTPUT_DESTINATION_PREFIX = f"root://eosuser.cern.ch//{EOS_PROJECT_ROOT.lstrip('/')}/out/condor_runs/"


class CondorTrainingTests(unittest.TestCase):
    def test_lightweight_test_preset_sets_one_epoch_and_espresso(self):
        class Args:
            condor_lightweight_test = True
            submit_training_to_condor = False
            train_best_model = False
            n_epochs = None
            condor_job_flavour = None
            condor_tag = None

        args = Args()
        apply_lightweight_test_preset(args)
        self.assertTrue(args.submit_training_to_condor)
        self.assertTrue(args.train_best_model)
        self.assertEqual(args.n_epochs, 1)
        self.assertEqual(args.condor_job_flavour, "espresso")
        self.assertEqual(args.condor_tag, "lightweight-test")

    def test_wrapper_contains_environment_bootstrap_and_epoch_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            spec = build_job_spec(
                repo_root=base / "repo",
                out_path=base / "out",
                input_path=base / "out",
                training_config_path=base / "cfg" / "training_config.yaml",
                tag="smoke",
                n_epochs=1,
            )
            wrapper = render_wrapper_script(spec)
            self.assertEqual(spec.environment_manager, "mamba")
            self.assertEqual(spec.environment_name, "HHbbgg_classifier")
            self.assertIn("micromamba activate HHbbgg_classifier", wrapper)
            self.assertIn("mamba activate HHbbgg_classifier", wrapper)
            self.assertIn("scramv1 runtime -sh", wrapper)
            self.assertIn("--n_epochs 1", wrapper)
            self.assertIn("models/training_utils.py", wrapper)
            self.assertIn("export HHBBGG_CONDOR_MODE=1", wrapper)

    def test_environment_config_is_loaded_from_run_config_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg_dir = base / "cfg"
            cfg_dir.mkdir()
            (cfg_dir / "training_config.yaml").write_text("do_random_search: false\n", encoding="utf-8")
            (cfg_dir / "environment.yaml").write_text(
                "manager: conda\nname: custom-env\n",
                encoding="utf-8",
            )
            env_config = load_environment_config(cfg_dir / "training_config.yaml")
            self.assertEqual(env_config, {"manager": "conda", "name": "custom-env"})

            spec = build_job_spec(
                repo_root=base / "repo",
                out_path=base / "out",
                input_path=base / "out",
                training_config_path=cfg_dir / "training_config.yaml",
            )
            wrapper = render_wrapper_script(spec)
            self.assertEqual(spec.environment_manager, "conda")
            self.assertEqual(spec.environment_name, "custom-env")
            self.assertIn("conda activate custom-env", wrapper)
            self.assertNotIn("micromamba activate custom-env", wrapper)

    def test_submit_file_requests_gpu_and_logs_under_run_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            spec = build_job_spec(
                repo_root=base / "repo",
                out_path=base / "out",
                input_path=base / "out",
                training_config_path=base / "cfg" / "training_config.yaml",
                tag="gpu",
                gpus=2,
                cpus=8,
                memory_gb=64,
                disk_gb=50,
                accounting_group="group_cms.test",
                job_flavour="tomorrow",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                submit_text = render_submit_file(spec)
            self.assertIn("request_gpus = 2", submit_text)
            self.assertIn("request_cpus = 8", submit_text)
            self.assertIn("request_memory = 64 GB", submit_text)
            self.assertIn("accounting_group = group_cms.test", submit_text)
            self.assertIn('+JobFlavour = "tomorrow"', submit_text)
            self.assertIn("log = train_$(ClusterId).log", submit_text)
            self.assertIn("output = train_$(ClusterId).$(Process).out", submit_text)
            self.assertIn("error = train_$(ClusterId).$(Process).err", submit_text)

    def test_spool_submit_file_sets_output_destination_for_eos(self):
        spec = build_job_spec(
            repo_root=EOS_REPO_ROOT,
            out_path=EOS_OUT_PATH,
            input_path=EOS_INPUT_PATH,
            training_config_path=EOS_TRAINING_CONFIG_PATH,
            submission_mode="spool",
        )
        submit_text = render_submit_file(spec)
        self.assertIn("log = train_$(ClusterId).log", submit_text)
        self.assertIn("output = train_$(ClusterId).$(Process).out", submit_text)
        self.assertIn("error = train_$(ClusterId).$(Process).err", submit_text)
        self.assertIn(EOS_OUTPUT_DESTINATION_PREFIX, submit_text)
        self.assertIn("MY.XRDCP_CREATE_DIR = True", submit_text)

    def test_default_condor_root_is_nested_under_out_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            spec = build_job_spec(
                repo_root=base / "repo",
                out_path=base / "training_out",
                input_path=base / "training_out",
                training_config_path=base / "cfg" / "training_config.yaml",
            )
            self.assertEqual(spec.condor_root.parent, base / "training_out" / "condor_runs")

    def test_submit_log_parser_extracts_schedd_alias(self):
        log_text = "000 (...) Job submitted from host: <...&alias=bigbird24.cern.ch&...>"
        self.assertEqual(parse_submit_schedd_from_log(log_text), "bigbird24.cern.ch")

    def test_explicit_schedd_is_stored_in_job_spec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            spec = build_job_spec(
                repo_root=base / "repo",
                out_path=base / "training_out",
                input_path=base / "training_out",
                training_config_path=base / "cfg" / "training_config.yaml",
                schedd="bigbird24.cern.ch",
            )
            self.assertEqual(spec.schedd, "bigbird24.cern.ch")

    def test_standard_schedd_is_rejected_for_eos_workspace(self):
        spec = build_job_spec(
            repo_root=EOS_REPO_ROOT,
            out_path=EOS_OUT_PATH,
            input_path=EOS_INPUT_PATH,
            training_config_path=EOS_TRAINING_CONFIG_PATH,
            schedd="bigbird19.cern.ch",
            submission_mode="eossubmit",
        )
        with self.assertRaises(ValueError):
            validate_submit_target(spec)

    def test_standard_schedd_is_allowed_for_eos_workspace_in_spool_mode(self):
        spec = build_job_spec(
            repo_root=EOS_REPO_ROOT,
            out_path=EOS_OUT_PATH,
            input_path=EOS_INPUT_PATH,
            training_config_path=EOS_TRAINING_CONFIG_PATH,
            schedd="bigbird19.cern.ch",
            submission_mode="spool",
        )
        validate_submit_target(spec)

    def test_build_submit_command_uses_spool_mode_by_default(self):
        spec = build_job_spec(
            repo_root="/tmp/repo",
            out_path="/tmp/out",
            input_path="/tmp/out",
            training_config_path="/tmp/cfg/training_config.yaml",
        )
        cmd = build_submit_command(spec)
        self.assertEqual(cmd[0:2], ["bash", "-lc"])
        self.assertIn("module load lxbatch/share", cmd[2])
        self.assertIn("condor_submit -spool", cmd[2])
        self.assertNotIn("module load lxbatch/eossubmit", cmd[2])

    def test_build_submit_command_uses_eossubmit_mode_when_requested(self):
        spec = build_job_spec(
            repo_root="/tmp/repo",
            out_path="/tmp/out",
            input_path="/tmp/out",
            training_config_path="/tmp/cfg/training_config.yaml",
            submission_mode="eossubmit",
        )
        cmd = build_submit_command(spec)
        self.assertIn("module load lxbatch/eossubmit", cmd[2])
        self.assertNotIn("condor_submit -spool", cmd[2])

    def test_machine_constraint_contains_resource_requests(self):
        spec = build_job_spec(
            repo_root="/tmp/repo",
            out_path="/tmp/out",
            input_path="/tmp/out",
            training_config_path="/tmp/cfg/training_config.yaml",
            cpus=4,
            memory_gb=32,
            disk_gb=20,
            gpus=1,
            requirements='OpSysAndVer == "AlmaLinux9"',
        )
        constraint = build_machine_constraint(spec)
        self.assertIn("TotalCpus", constraint)
        self.assertIn("TotalMemory", constraint)
        self.assertIn("TotalDisk", constraint)
        self.assertIn("TotalGpus", constraint)
        self.assertIn('OpSysAndVer == "AlmaLinux9"', constraint)

    @patch("submission.condor_training.subprocess.run")
    def test_better_analyze_uses_schedd_when_provided(self, mock_run):
        mock_run.return_value.stdout = "analysis"
        mock_run.return_value.stderr = ""
        output = better_analyze_job("124674", schedd="bigbird24.cern.ch")
        self.assertEqual(output, "analysis")
        cmd = mock_run.call_args.args[0]
        self.assertEqual(cmd, ["condor_q", "-name", "bigbird24.cern.ch", "124674", "-better-analyze"])


if __name__ == "__main__":
    unittest.main()
