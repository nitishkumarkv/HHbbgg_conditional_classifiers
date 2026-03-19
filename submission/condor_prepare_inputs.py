from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from submission.condor_training import (
    build_submit_command,
    job_flavor_to_max_runtime_seconds,
    load_environment_config,
    parse_submit_schedd_from_log,
)


@dataclass
class CondorPrepareJobSpec:
    repo_root: Path
    out_path: Path
    config_path: Path
    training_config_path: Path
    condor_root: Path
    run_id: str
    log_dir: Path
    wrapper_path: Path
    submit_path: Path
    metadata_path: Path
    environment_manager: str = "mamba"
    environment_name: str = "HHbbgg_classifier"
    cpus: int = 4
    memory_gb: int = 32
    disk_gb: int = 20
    gpus: int = 1
    accounting_group: Optional[str] = None
    max_input_files: Optional[int] = None
    max_rows_per_file: Optional[int] = None
    job_flavor: Optional[str] = "testmatch"
    requirements: Optional[str] = None
    schedd: Optional[str] = None
    submission_mode: str = "spool"
    prep_inputs_for_training: bool = False
    prepare_inputs_pred_sim: bool = False
    prepare_inputs_pred_data: bool = False
    prepare_inputs_pred_sys: bool = False
    mhh_var: Optional[str] = None
    mhh_range: Optional[tuple[float, float]] = None


def make_prepare_run_id(tag: Optional[str] = None) -> str:
    from datetime import datetime
    from uuid import uuid4

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:8]
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag.strip()) if tag else "prepare"
    safe_tag = safe_tag.strip("-") or "prepare"
    return f"{timestamp}_{safe_tag}_{suffix}"


def build_prepare_job_spec(
    repo_root: str | Path,
    out_path: str | Path,
    config_path: str | Path,
    training_config_path: str | Path,
    tag: Optional[str] = None,
    condor_work_dir: Optional[str | Path] = None,
    cpus: int = 4,
    memory_gb: int = 32,
    disk_gb: int = 20,
    gpus: int = 1,
    max_input_files: Optional[int] = None,
    max_rows_per_file: Optional[int] = None,
    accounting_group: Optional[str] = None,
    job_flavor: Optional[str] = "testmatch",
    requirements: Optional[str] = None,
    schedd: Optional[str] = None,
    submission_mode: str = "spool",
    prep_inputs_for_training: bool = False,
    prepare_inputs_pred_sim: bool = False,
    prepare_inputs_pred_data: bool = False,
    prepare_inputs_pred_sys: bool = False,
    mhh_var: Optional[str] = None,
    mhh_range: Optional[tuple[float, float]] = None,
) -> CondorPrepareJobSpec:
    repo_root = Path(repo_root).resolve()
    out_path = Path(out_path).resolve()
    config_path = Path(config_path).resolve()
    training_config_path = Path(training_config_path).resolve()
    env_config = load_environment_config(training_config_path)
    run_id = make_prepare_run_id(tag)
    condor_root = Path(condor_work_dir).resolve() if condor_work_dir else (out_path / "condor_runs")
    run_dir = condor_root / run_id
    log_dir = run_dir / "logs"
    return CondorPrepareJobSpec(
        repo_root=repo_root,
        out_path=out_path,
        config_path=config_path,
        training_config_path=training_config_path,
        condor_root=run_dir,
        run_id=run_id,
        log_dir=log_dir,
        wrapper_path=run_dir / "prepare_wrapper.sh",
        submit_path=run_dir / "prepare_job.sub",
        metadata_path=run_dir / "job_metadata.json",
        environment_manager=env_config["manager"],
        environment_name=env_config["name"],
        cpus=cpus,
        memory_gb=memory_gb,
        disk_gb=disk_gb,
        gpus=gpus,
        max_input_files=max_input_files,
        max_rows_per_file=max_rows_per_file,
        accounting_group=accounting_group,
        job_flavor=job_flavor,
        requirements=requirements,
        schedd=schedd,
        submission_mode=submission_mode,
        prep_inputs_for_training=prep_inputs_for_training,
        prepare_inputs_pred_sim=prepare_inputs_pred_sim,
        prepare_inputs_pred_data=prepare_inputs_pred_data,
        prepare_inputs_pred_sys=prepare_inputs_pred_sys,
        mhh_var=mhh_var,
        mhh_range=mhh_range,
    )


def _environment_activation_commands(spec: CondorPrepareJobSpec) -> list[str]:
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


def _prepare_command_parts(spec: CondorPrepareJobSpec) -> list[str]:
    args = [
        "python3",
        "run_prepare_inputs.py",
        "--config_path",
        str(spec.config_path),
        "--out_path",
        str(spec.out_path),
    ]
    if spec.prep_inputs_for_training:
        args.append("--prep_inputs_for_training")
    if spec.prepare_inputs_pred_sim:
        args.append("--prepare_inputs_pred_sim")
    if spec.prepare_inputs_pred_data:
        args.append("--prepare_inputs_pred_data")
    if spec.prepare_inputs_pred_sys:
        args.append("--prepare_inputs_pred_sys")
    if spec.max_input_files is not None:
        args.extend(["--max_input_files", str(spec.max_input_files)])
    if spec.max_rows_per_file is not None:
        args.extend(["--max_rows_per_file", str(spec.max_rows_per_file)])
    if spec.mhh_var is not None:
        args.extend(["--mhh_var", spec.mhh_var])
    if spec.mhh_range is not None:
        lo, hi = spec.mhh_range
        args.extend(["--mhh_min", str(lo), "--mhh_max", str(hi)])
    return args


def render_wrapper_script(spec: CondorPrepareJobSpec) -> str:
    prepare_cmd = " ".join(shlex.quote(part) for part in _prepare_command_parts(spec))
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            'echo "INFO: Starting Condor prepare wrapper on $(hostname) at $(date --iso-8601=seconds)"',
            f'echo "INFO: Repo root: {shlex.quote(str(spec.repo_root))}"',
            f'echo "INFO: Config path: {shlex.quote(str(spec.config_path))}"',
            f'echo "INFO: Output path: {shlex.quote(str(spec.out_path))}"',
            f'echo "INFO: Training config path: {shlex.quote(str(spec.training_config_path))}"',
            f'echo "INFO: mHH var: {shlex.quote(spec.mhh_var or "")}"',
            f'echo "INFO: mHH range: {spec.mhh_range if spec.mhh_range is not None else ()}"',
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
            *(_environment_activation_commands(spec)),
            "",
            f"cd {shlex.quote(str(spec.repo_root))}",
            "export HHBBGG_CONDOR_MODE=1",
            'echo "INFO: Python executable: $(command -v python3)"',
            f"echo \"INFO: Executing: {prepare_cmd}\"",
            prepare_cmd,
        ]
    ) + "\n"


def _build_output_destination(spec: CondorPrepareJobSpec) -> Optional[str]:
    if spec.submission_mode != "spool":
        return None

    path = str(spec.log_dir)
    if path.startswith("/eos/user/"):
        return f"root://eosuser.cern.ch//{path.lstrip('/')}/"

    home_match = re.match(r"^/eos/home-([A-Za-z])/([^/]+)(/.*)?$", path)
    if home_match:
        initial = home_match.group(1).lower()
        username = home_match.group(2)
        suffix = home_match.group(3) or ""
        return f"root://eosuser.cern.ch//eos/user/{initial}/{username}{suffix}/"

    return None


def _build_log_paths() -> tuple[str, str, str]:
    return (
        "prepare_$(ClusterId).log",
        "prepare_$(ClusterId).$(Process).out",
        "prepare_$(ClusterId).$(Process).err",
    )


def render_submit_file(spec: CondorPrepareJobSpec) -> str:
    output_destination = _build_output_destination(spec)
    log_path, stdout_path, stderr_path = _build_log_paths()
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
    if spec.job_flavor:
        lines.append(f'+JobFlavor = "{spec.job_flavor}"')
        lines.append(f"+MaxRuntime = {job_flavor_to_max_runtime_seconds(spec.job_flavor)}")
    if spec.requirements:
        lines.append(f"requirements = ({spec.requirements})")
    lines.append("queue 1")
    return "\n".join(lines) + "\n"


def ensure_job_files(spec: CondorPrepareJobSpec) -> CondorPrepareJobSpec:
    spec.condor_root.mkdir(parents=True, exist_ok=True)
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    spec.wrapper_path.write_text(render_wrapper_script(spec), encoding="utf-8")
    os.chmod(spec.wrapper_path, 0o755)
    spec.submit_path.write_text(render_submit_file(spec), encoding="utf-8")
    spec.metadata_path.write_text(json.dumps(_metadata_dict(spec), indent=2), encoding="utf-8")
    return spec


def _metadata_dict(spec: CondorPrepareJobSpec) -> dict[str, object]:
    data = asdict(spec)
    return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}


def get_log_path_for_cluster(spec: CondorPrepareJobSpec, cluster_id: str) -> Path:
    return spec.log_dir / f"prepare_{cluster_id}.log"


def _wait_for_submit_schedd(
    spec: CondorPrepareJobSpec,
    cluster_id: Optional[str],
    timeout_seconds: float = 2.0,
) -> Optional[str]:
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


def submit_prepare_job(spec: CondorPrepareJobSpec, dry_run: bool = False) -> dict[str, Optional[str]]:
    if dry_run:
        ensure_job_files(spec)
        return {"cluster_id": None, "stdout": None, "stderr": None, "schedd": None}

    ensure_job_files(spec)
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

    cluster_id = re.search(r"cluster\s+(\d+)", result.stdout, re.IGNORECASE)
    cluster_id = cluster_id.group(1) if cluster_id else None
    schedd = _wait_for_submit_schedd(spec, cluster_id)
    return {
        "cluster_id": cluster_id,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "schedd": schedd,
    }
