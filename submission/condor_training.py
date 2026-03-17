from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4
import yaml


@dataclass
class CondorJobSpec:
    repo_root: Path
    input_path: Path
    training_config_path: Path
    condor_root: Path
    run_id: str
    log_dir: Path
    wrapper_path: Path
    submit_path: Path
    metadata_path: Path
    out_path: Path
    environment_manager: str = "mamba"
    environment_name: str = "HHbbgg_classifier"
    cpus: int = 4
    memory_gb: int = 32
    disk_gb: int = 20
    gpus: int = 1
    accounting_group: Optional[str] = None
    job_flavour: Optional[str] = None
    requirements: Optional[str] = None
    n_epochs: Optional[int] = None
    schedd: Optional[str] = None
    submission_mode: str = "spool"


def make_run_id(tag: Optional[str] = None) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:8]
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag.strip()) if tag else "train"
    safe_tag = safe_tag.strip("-") or "train"
    return f"{timestamp}_{safe_tag}_{suffix}"


def build_job_spec(
    repo_root: str | Path,
    out_path: str | Path,
    input_path: str | Path,
    training_config_path: str | Path,
    tag: Optional[str] = None,
    condor_work_dir: Optional[str | Path] = None,
    cpus: int = 4,
    memory_gb: int = 32,
    disk_gb: int = 20,
    gpus: int = 1,
    accounting_group: Optional[str] = None,
    job_flavour: Optional[str] = None,
    requirements: Optional[str] = None,
    n_epochs: Optional[int] = None,
    schedd: Optional[str] = None,
    submission_mode: str = "spool",
) -> CondorJobSpec:
    repo_root = Path(repo_root).resolve()
    out_path = Path(out_path).resolve()
    input_path = Path(input_path).resolve()
    training_config_path = Path(training_config_path).resolve()
    env_config = load_environment_config(training_config_path)
    run_id = make_run_id(tag)
    condor_root = Path(condor_work_dir).resolve() if condor_work_dir else (out_path / "condor_runs")
    run_dir = condor_root / run_id
    log_dir = run_dir / "logs"
    return CondorJobSpec(
        repo_root=repo_root,
        input_path=input_path,
        training_config_path=training_config_path,
        condor_root=run_dir,
        run_id=run_id,
        log_dir=log_dir,
        wrapper_path=run_dir / "train_wrapper.sh",
        submit_path=run_dir / "train_job.sub",
        metadata_path=run_dir / "job_metadata.json",
        out_path=out_path,
        environment_manager=env_config["manager"],
        environment_name=env_config["name"],
        cpus=cpus,
        memory_gb=memory_gb,
        disk_gb=disk_gb,
        gpus=gpus,
        accounting_group=accounting_group,
        job_flavour=job_flavour,
        requirements=requirements,
        n_epochs=n_epochs,
        schedd=schedd,
        submission_mode=submission_mode,
    )


def load_environment_config(training_config_path: str | Path) -> Dict[str, str]:
    defaults = {"manager": "mamba", "name": "HHbbgg_classifier"}
    config_path = Path(training_config_path).resolve().parent / "environment.yaml"
    if not config_path.exists():
        return defaults

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    manager = str(config.get("manager", config.get("environment_manager", defaults["manager"]))).strip().lower()
    name = str(config.get("name", config.get("environment_name", defaults["name"]))).strip()
    if manager == "micromamba":
        manager = "mamba"
    if manager not in {"mamba", "conda"}:
        raise ValueError(
            f"Unsupported environment manager '{manager}' in {config_path}. "
            "Supported values are 'mamba' and 'conda'."
        )
    if not name:
        raise ValueError(f"Environment name in {config_path} cannot be empty.")
    return {"manager": manager, "name": name}


def render_environment_activation(spec: CondorJobSpec) -> list[str]:
    if spec.environment_manager == "conda":
        return [
            "if command -v conda >/dev/null 2>&1; then",
            "  source \"$(conda info --base)/etc/profile.d/conda.sh\"",
            f"  conda activate {shlex.quote(spec.environment_name)}",
            "else",
            '  echo "ERROR: Requested environment manager conda, but conda is not available." >&2',
            "  exit 1",
            "fi",
        ]

    return [
        "if command -v micromamba >/dev/null 2>&1; then",
        "  eval \"$(micromamba shell hook -s bash)\"",
        f"  micromamba activate {shlex.quote(spec.environment_name)}",
        "elif command -v mamba >/dev/null 2>&1; then",
        "  eval \"$(mamba shell hook --shell bash)\"",
        f"  mamba activate {shlex.quote(spec.environment_name)}",
        "elif command -v conda >/dev/null 2>&1; then",
        "  source \"$(conda info --base)/etc/profile.d/conda.sh\"",
        f"  conda activate {shlex.quote(spec.environment_name)}",
        "else",
        '  echo "ERROR: Requested environment manager mamba, but neither micromamba, mamba, nor conda is available." >&2',
        "  exit 1",
        "fi",
    ]


def render_wrapper_script(spec: CondorJobSpec) -> str:
    cmd = [
        "python3",
        "models/training_utils.py",
        "--input_path",
        str(spec.input_path),
        "--training_config_path",
        str(spec.training_config_path),
    ]
    if spec.n_epochs is not None:
        cmd.extend(["--n_epochs", str(spec.n_epochs)])

    training_cmd = " ".join(shlex.quote(part) for part in cmd)

    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            'echo "INFO: Starting Condor training wrapper on $(hostname) at $(date --iso-8601=seconds)"',
            f'echo "INFO: Repo root: {shlex.quote(str(spec.repo_root))}"',
            f'echo "INFO: Training input path: {shlex.quote(str(spec.input_path))}"',
            f'echo "INFO: Training config path: {shlex.quote(str(spec.training_config_path))}"',
            f'echo "INFO: Output path: {shlex.quote(str(spec.out_path))}"',
            "",
            "if [ -f /cvmfs/cms.cern.ch/cmsset_default.sh ]; then",
            "  # EOS/lxplus submissions commonly require a CMS environment via scram.",
            "  source /cvmfs/cms.cern.ch/cmsset_default.sh",
            "fi",
            "",
            "if [ -n \"${CMSSW_BASE:-}\" ]; then",
            "  if command -v scramv1 >/dev/null 2>&1; then",
            "    eval \"$(scramv1 runtime -sh)\"",
            "  elif command -v scram >/dev/null 2>&1; then",
            "    eval \"$(scram runtime -sh)\"",
            "  fi",
            "fi",
            "",
            *render_environment_activation(spec),
            "",
            f"cd {shlex.quote(str(spec.repo_root))}",
            "export HHBBGG_CONDOR_MODE=1",
            'echo "INFO: Python executable: $(command -v python3)"',
            'echo "INFO: CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"',
            f"echo \"INFO: Executing: {training_cmd}\"",
            training_cmd,
        ]
    ) + "\n"


def render_submit_file(spec: CondorJobSpec) -> str:
    output_destination = _build_output_destination(spec)
    log_path, stdout_path, stderr_path = _build_log_paths(spec)
    lines = [
        "universe = vanilla",
        f"executable = {spec.wrapper_path}",
        "arguments =",
        "getenv = True",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT",
        f"transfer_input_files = {spec.wrapper_path}",
        "transfer_output_files =",
        "transfer_executable = True",
        f"initialdir = {spec.condor_root}",
        f"log = {log_path}",
        f"output = {stdout_path}",
        f"error = {stderr_path}",
        f"request_cpus = {spec.cpus}",
        f"request_memory = {spec.memory_gb} GB",
        f"request_disk = {spec.disk_gb} GB",
    ]
    if output_destination:
        lines.append(f"output_destination = {output_destination}")
        lines.append("MY.XRDCP_CREATE_DIR = True")
    if spec.gpus > 0:
        lines.append(f"request_gpus = {spec.gpus}")
    if spec.accounting_group:
        lines.append(f"accounting_group = {spec.accounting_group}")
    if spec.job_flavour:
        lines.append(f'+JobFlavour = "{spec.job_flavour}"')
    if spec.requirements:
        lines.append(f"requirements = {spec.requirements}")
    lines.append("queue 1")
    return "\n".join(lines) + "\n"


def ensure_job_files(spec: CondorJobSpec) -> CondorJobSpec:
    spec.condor_root.mkdir(parents=True, exist_ok=True)
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    spec.wrapper_path.write_text(render_wrapper_script(spec), encoding="utf-8")
    os.chmod(spec.wrapper_path, 0o755)
    spec.submit_path.write_text(render_submit_file(spec), encoding="utf-8")
    spec.metadata_path.write_text(json.dumps(_metadata_dict(spec), indent=2), encoding="utf-8")
    return spec


def validate_submit_target(spec: CondorJobSpec) -> None:
    eos_paths = [spec.condor_root, spec.submit_path, spec.wrapper_path]
    uses_eos = any(str(path).startswith("/eos/") for path in eos_paths)
    if uses_eos and spec.submission_mode == "eossubmit" and spec.schedd and "eos" not in spec.schedd.lower():
        raise ValueError(
            "The Condor workspace is on EOS, but --condor_submission_mode=eossubmit "
            f"was paired with a standard schedd ({spec.schedd}). "
            "Use your default routing, switch to --condor_submission_mode=spool, "
            "or target an EosSubmit schedd explicitly."
        )


def build_machine_constraint(spec: CondorJobSpec, only_available: bool = False) -> str:
    clauses = [
        "Arch =!= undefined",
        f"((TotalCpus =!= undefined && TotalCpus >= {spec.cpus}) || (Cpus =!= undefined && Cpus >= {spec.cpus}))",
        f"((TotalMemory =!= undefined && TotalMemory >= {spec.memory_gb * 1024}) || (Memory =!= undefined && Memory >= {spec.memory_gb * 1024}))",
        f"((TotalDisk =!= undefined && TotalDisk >= {spec.disk_gb * 1024 * 1024}) || (Disk =!= undefined && Disk >= {spec.disk_gb * 1024 * 1024}))",
    ]
    if spec.gpus > 0:
        clauses.append(f"((TotalGpus =!= undefined && TotalGpus >= {spec.gpus}) || (Gpus =!= undefined && Gpus >= {spec.gpus}))")
    if spec.requirements:
        clauses.append(f"({spec.requirements})")
    if only_available:
        clauses.extend(['State == "Unclaimed"', 'Activity == "Idle"'])
    return " && ".join(clauses)


def _run_condor_status_json(constraint: str) -> list[dict]:
    cmd = ["condor_status", "-json", "-constraint", constraint]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    text = result.stdout.strip()
    if not text:
        return []
    return json.loads(text)


def summarize_resource_matches(spec: CondorJobSpec) -> Dict[str, object]:
    capable_ads = _run_condor_status_json(build_machine_constraint(spec, only_available=False))
    available_ads = _run_condor_status_json(build_machine_constraint(spec, only_available=True))

    def machine_count(ads: list[dict]) -> int:
        return len({ad.get("Machine", ad.get("Name")) for ad in ads})

    def sample_names(ads: list[dict], limit: int = 5) -> list[str]:
        names = []
        for ad in ads[:limit]:
            names.append(str(ad.get("Machine", ad.get("Name", "<unknown>"))))
        return names

    return {
        "capable_slots": len(capable_ads),
        "capable_machines": machine_count(capable_ads),
        "available_slots": len(available_ads),
        "available_machines": machine_count(available_ads),
        "capable_examples": sample_names(capable_ads),
        "available_examples": sample_names(available_ads),
        "constraint": build_machine_constraint(spec, only_available=False),
        "available_constraint": build_machine_constraint(spec, only_available=True),
    }


def better_analyze_job(cluster_id: str, schedd: Optional[str] = None) -> str:
    cmd = ["condor_q"]
    if schedd:
        cmd.extend(["-name", schedd])
    cmd.extend([cluster_id, "-better-analyze"])
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return result.stdout


def _metadata_dict(spec: CondorJobSpec) -> Dict[str, object]:
    data = asdict(spec)
    return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}


def _build_output_destination(spec: CondorJobSpec) -> Optional[str]:
    if spec.submission_mode != "spool":
        return None
    eos_path = _as_eos_xrootd_path(spec.log_dir)
    if eos_path is None:
        warnings.warn(
            "Spool submission requested, but the Condor log directory is not on EOS. "
            "No EOS output_destination could be derived.",
            stacklevel=2,
        )
        return None
    return f"root://eosuser.cern.ch//{eos_path.lstrip('/')}/"


def _build_log_paths(spec: CondorJobSpec) -> tuple[str, str, str]:
    if spec.submission_mode == "spool":
        return (
            "train_$(ClusterId).log",
            "train_$(ClusterId).$(Process).out",
            "train_$(ClusterId).$(Process).err",
        )
    return (
        f"{spec.log_dir}/train_$(ClusterId).log",
        f"{spec.log_dir}/train_$(ClusterId).$(Process).out",
        f"{spec.log_dir}/train_$(ClusterId).$(Process).err",
    )


def _as_eos_xrootd_path(path: Path) -> Optional[str]:
    path_str = str(path)
    if path_str.startswith("/eos/user/"):
        return path_str
    home_match = re.match(r"^/eos/home-([A-Za-z])/([^/]+)(/.*)?$", path_str)
    if home_match:
        initial = home_match.group(1).lower()
        username = home_match.group(2)
        suffix = home_match.group(3) or ""
        return f"/eos/user/{initial}/{username}{suffix}"
    return None


def build_submit_command(spec: CondorJobSpec) -> list[str]:
    submit_cmd = "condor_submit"
    if spec.schedd:
        submit_cmd += f" -name {shlex.quote(spec.schedd)}"
    if spec.submission_mode == "spool":
        submit_cmd += " -spool"
    submit_cmd += f" {shlex.quote(str(spec.submit_path))}"
    bash_parts = [
        "source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1 || true",
        "if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi",
    ]
    if spec.submission_mode == "eossubmit":
        bash_parts.append("if command -v module >/dev/null 2>&1; then module load lxbatch/eossubmit >/dev/null 2>&1 || true; fi")
    if spec.submission_mode == "spool":
        bash_parts.append("if command -v module >/dev/null 2>&1; then module load lxbatch/share >/dev/null 2>&1 || true; fi")
    bash_parts.extend(
        [
            "if [ -n \"${CMSSW_BASE:-}\" ]; then "
            "if command -v scramv1 >/dev/null 2>&1; then eval \"$(scramv1 runtime -sh)\"; "
            "elif command -v scram >/dev/null 2>&1; then eval \"$(scram runtime -sh)\"; fi; "
            "fi",
            submit_cmd,
        ]
    )
    bash_cmd = "; ".join(bash_parts)
    return ["bash", "-lc", bash_cmd]


def parse_cluster_id(stdout: str) -> Optional[str]:
    match = re.search(r"cluster\s+(\d+)", stdout, re.IGNORECASE)
    return match.group(1) if match else None


def get_log_path_for_cluster(spec: CondorJobSpec, cluster_id: str) -> Path:
    return spec.log_dir / f"train_{cluster_id}.log"


def parse_submit_schedd_from_log(log_text: str) -> Optional[str]:
    match = re.search(r"alias=([A-Za-z0-9._-]+)", log_text)
    return match.group(1) if match else None


def wait_for_submit_schedd(spec: CondorJobSpec, cluster_id: Optional[str], timeout_seconds: float = 2.0) -> Optional[str]:
    if not cluster_id:
        return None

    log_path = get_log_path_for_cluster(spec, cluster_id)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="ignore")
            schedd = parse_submit_schedd_from_log(log_text)
            if schedd:
                return schedd
        time.sleep(0.1)
    return None


def submit_job(spec: CondorJobSpec, dry_run: bool = False) -> Dict[str, Optional[str]]:
    validate_submit_target(spec)
    ensure_job_files(spec)
    if dry_run:
        return {"cluster_id": None, "stdout": None, "stderr": None, "schedd": None}

    try:
        result = subprocess.run(build_submit_command(spec), check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        raise RuntimeError(
            "condor_submit failed.\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        ) from exc
    cluster_id = parse_cluster_id(result.stdout)
    schedd = wait_for_submit_schedd(spec, cluster_id)
    return {"cluster_id": cluster_id, "stdout": result.stdout, "stderr": result.stderr, "schedd": schedd}


def apply_lightweight_test_preset(args) -> None:
    args.submit_training_to_condor = True
    args.train_best_model = True
    args.n_epochs = 1
    args.condor_job_flavour = "espresso"
    if getattr(args, "condor_tag", None) is None:
        args.condor_tag = "lightweight-test"
