import tempfile
import unittest
from pathlib import Path

from submission.condor_prepare_inputs import (
    build_prepare_job_spec,
    render_submit_file,
    render_wrapper_script,
    submit_prepare_job,
)


class CondorPrepareInputsTests(unittest.TestCase):
    def test_prepare_submit_default_job_flavor_testmatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            spec = build_prepare_job_spec(
                repo_root=base / "repo",
                out_path=base / "out",
                config_path=base / "cfg",
                training_config_path=base / "cfg" / "training_config.yaml",
                tag="prepare",
                prep_inputs_for_training=True,
            )
            self.assertEqual(spec.job_flavor, "testmatch")
            submit_text = render_submit_file(spec)
            self.assertIn('+JobFlavor = "testmatch"', submit_text)
            self.assertIn("+MaxRuntime = 259200", submit_text)

    def test_prepare_wrapper_prints_expected_elements(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            spec = build_prepare_job_spec(
                repo_root=base / "repo",
                out_path=base / "out",
                config_path=base / "cfg",
                training_config_path=base / "cfg" / "training_config.yaml",
                tag="prepare",
                prep_inputs_for_training=True,
                prepare_inputs_pred_sim=True,
                prepare_inputs_pred_data=True,
                prepare_inputs_pred_sys=True,
                mhh_var="mHH",
                mhh_range=(0.0, 350.0),
            )
            wrapper = render_wrapper_script(spec)
            self.assertIn("python3 run_prepare_inputs.py", wrapper)
            self.assertIn("--prep_inputs_for_training", wrapper)
            self.assertIn("--prepare_inputs_pred_sim", wrapper)
            self.assertIn("--prepare_inputs_pred_data", wrapper)
            self.assertIn("--prepare_inputs_pred_sys", wrapper)
            self.assertIn("--mhh_min 0.0", wrapper)
            self.assertIn("--mhh_max 350.0", wrapper)

            self.assertNotIn("--max_input_files", wrapper)
            self.assertNotIn("--max_rows_per_file", wrapper)

    def test_prepare_submit_file_requests_requested_resources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            spec = build_prepare_job_spec(
                repo_root=base / "repo",
                out_path=base / "out",
                config_path=base / "cfg",
                training_config_path=base / "cfg" / "training_config.yaml",
                cpus=8,
                memory_gb=64,
                disk_gb=50,
                gpus=2,
                accounting_group="group_cms.test",
                job_flavor="tomorrow",
                prep_inputs_for_training=True,
            )
            submit_text = render_submit_file(spec)
            self.assertIn("request_gpus = 2", submit_text)
            self.assertIn("request_cpus = 8", submit_text)
            self.assertIn("request_memory = 64 GB", submit_text)
            self.assertIn("accounting_group = group_cms.test", submit_text)
            self.assertIn('+JobFlavor = "tomorrow"', submit_text)
            self.assertIn("+MaxRuntime = 86400", submit_text)
            self.assertIn("log = prepare_$(ClusterId).log", submit_text)
            self.assertIn("output = prepare_$(ClusterId).$(Process).out", submit_text)
            self.assertIn("error = prepare_$(ClusterId).$(Process).err", submit_text)

    def test_prepare_wrapper_includes_lightweight_caps_when_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            spec = build_prepare_job_spec(
                repo_root=base / "repo",
                out_path=base / "out",
                config_path=base / "cfg",
                training_config_path=base / "cfg" / "training_config.yaml",
                prep_inputs_for_training=True,
                max_input_files=2,
                max_rows_per_file=5000,
            )
            wrapper = render_wrapper_script(spec)
            self.assertIn("--max_input_files 2", wrapper)
            self.assertIn("--max_rows_per_file 5000", wrapper)

    def test_submit_prepare_job_parses_cluster_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            spec = build_prepare_job_spec(
                repo_root=base / "repo",
                out_path=base / "out",
                config_path=base / "cfg",
                training_config_path=base / "cfg" / "training_config.yaml",
            )
            self.assertTrue(callable(submit_prepare_job))



if __name__ == "__main__":
    unittest.main()
