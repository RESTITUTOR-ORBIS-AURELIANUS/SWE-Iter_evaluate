#!/usr/bin/env python3
"""SWE-Iter v1 evaluator.

This is a formal first-pass implementation for Python repositories with pytest.
It clones a repository, constructs iterative PR steps from a mined PR chain,
classifies case-level tests, calls SWE-agent for code generation, and uses a
DeepSeek/OpenAI-compatible API for semantic PatchScore judging.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests
import yaml


class SWEIterError(RuntimeError):
    """Raised for expected evaluator failures."""


DEFAULT_CONFIG: dict[str, Any] = {
    "api": {
        "github": {"token": ""},
        "deepseek_pro": {
            "api_key": "",
            "base_url": "",
            "model": "DeepSeek-V4-Pro",
        },
    },
    "swe_agent": {
        "command": "sweagent",
        "config_path": "",
        "extra_args": [],
        "timeout_seconds_per_step": 3600,
        "apply_patch_locally": True,
        "command_template": [],
    },
    "scoring": {
        "lambda_test": 0.5,
        "gamma_time": 1.2,
        "rho_evidence": 0.5,
        "test_f2p_weight": 0.7,
        "test_p2p_weight": 0.3,
        "kappa_tests": 20,
    },
}

PATCH_SCORE_EXCLUDED_REQUIREMENT_TYPES = {"test", "docs", "non_source"}


@dataclass
class RuntimePaths:
    output_dir: Path
    patches_gold: Path
    patches_model: Path
    problem_statements: Path
    logs: Path


@dataclass
class PRStep:
    step_id: int
    pr_number: int
    title: str = ""
    body: str = ""
    commit_messages: list[str] = field(default_factory=list)
    merge_commit_message: str = ""
    from_commit: str = ""
    to_commit: str = ""
    requirement_source: str = "pull_request"
    requirement_description: str = ""
    golden_patch_file: str = ""
    patch_stats: dict[str, Any] = field(default_factory=dict)
    tests: dict[str, Any] = field(default_factory=dict)
    agent_result: dict[str, Any] = field(default_factory=dict)
    atomic_requirements: list[dict[str, Any]] = field(default_factory=list)
    patch_judgment: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, Any] = field(default_factory=dict)
    test_sources: dict[str, bytes] = field(default_factory=dict, repr=False)

    def as_result(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "pr_number": self.pr_number,
            "requirement": {
                "source": self.requirement_source,
                "title": self.title,
                "description": self.requirement_description,
                "commit_messages": self.commit_messages,
            },
            "from_commit": self.from_commit,
            "to_commit": self.to_commit,
            "golden_patch_file": self.golden_patch_file,
            "patch_stats": self.patch_stats,
            "tests": self.tests,
            "agent_result": self.agent_result,
            "atomic_requirements": self.atomic_requirements,
            "patch_judgment": self.patch_judgment,
            "scores": self.scores,
        }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path | None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path:
        path = Path(config_path)
        if path.exists():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            config = deep_merge(config, loaded)
        elif path.name != "config.yaml":
            raise SWEIterError(f"Error: config file not found: {path}")

    env_overrides = {
        ("api", "github", "token"): os.environ.get("GITHUB_TOKEN"),
        ("api", "deepseek_pro", "api_key"): os.environ.get("DEEPSEEK_V4_PRO_API_KEY"),
        ("api", "deepseek_pro", "base_url"): os.environ.get("DEEPSEEK_V4_PRO_BASE_URL"),
        ("swe_agent", "config_path"): os.environ.get("SWE_AGENT_CONFIG_PATH"),
        ("swe_agent", "command"): os.environ.get("SWE_AGENT_COMMAND"),
    }
    for path, value in env_overrides.items():
        if value:
            target = config
            for key in path[:-1]:
                target = target.setdefault(key, {})
            target[path[-1]] = value
    return config


def append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def run_command(
    cmd: list[str],
    cwd: str | Path | None = None,
    log_path: Path | None = None,
    timeout: int | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
    append: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if log_path and not append:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    if log_path:
        append_log(log_path, f"\n$ {' '.join(shlex.quote(part) for part in cmd)}")
        append_log(log_path, f"# cwd={Path(cwd).resolve() if cwd else Path.cwd()}")
        append_log(log_path, f"# started={started}")
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        if log_path:
            append_log(log_path, f"# timeout after {timeout}s")
            if exc.stdout:
                append_log(log_path, str(exc.stdout))
            if exc.stderr:
                append_log(log_path, str(exc.stderr))
        raise SWEIterError(f"Error: command timed out after {timeout}s: {cmd[0]}") from exc

    if log_path:
        if completed.stdout:
            append_log(log_path, completed.stdout)
        if completed.stderr:
            append_log(log_path, completed.stderr)
        append_log(log_path, f"# exit_code={completed.returncode}")

    if check and completed.returncode != 0:
        where = f" See log: {log_path}" if log_path else ""
        raise SWEIterError(
            f"Error: command failed with exit code {completed.returncode}: "
            f"{' '.join(shlex.quote(part) for part in cmd)}.{where}"
        )
    return completed


def ensure_dirs(output_dir: Path) -> RuntimePaths:
    paths = RuntimePaths(
        output_dir=output_dir,
        patches_gold=output_dir / "patches_gold",
        patches_model=output_dir / "patches_model",
        problem_statements=output_dir / "problem_statements",
        logs=output_dir / "logs",
    )
    for path in [
        paths.output_dir,
        paths.patches_gold,
        paths.patches_model,
        paths.problem_statements,
        paths.logs,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def repo_slug(repo: str) -> str:
    return repo.replace("/", "__")


class GitHubClient:
    def __init__(self, token: str | None, log_path: Path | None = None) -> None:
        self.token = token or ""
        self.log_path = log_path
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "swe-iter-eval-v1",
            }
        )
        if self.token and not self.token.startswith("FILL_"):
            self.session.headers["Authorization"] = f"Bearer {self.token}"

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        if self.log_path:
            append_log(self.log_path, f"GET {url}")
        response = self.session.get(url, params=params, timeout=60)
        if self.log_path:
            append_log(self.log_path, f"status={response.status_code}")
        if response.status_code >= 400:
            raise SWEIterError(
                f"Error: GitHub API request failed ({response.status_code}) for {url}"
            )
        return response.json(), response.links

    def paginated_get(self, url: str, params: dict[str, Any] | None = None) -> list[Any]:
        items: list[Any] = []
        current = url
        current_params = params or {}
        while current:
            payload, links = self._get(current, params=current_params)
            if isinstance(payload, list):
                items.extend(payload)
            else:
                items.append(payload)
            current = links.get("next", {}).get("url", "")
            current_params = {}
        return items

    def repo_api(self, repo: str, suffix: str) -> str:
        return f"https://api.github.com/repos/{repo}{suffix}"

    def fetch_pr_details(self, repo: str, pr_number: int) -> dict[str, Any]:
        payload, _ = self._get(self.repo_api(repo, f"/pulls/{pr_number}"))
        return payload

    def fetch_pr_commits(self, repo: str, pr_number: int) -> list[dict[str, Any]]:
        return self.paginated_get(
            self.repo_api(repo, f"/pulls/{pr_number}/commits"),
            params={"per_page": 100},
        )

    def fetch_commit(self, repo: str, sha: str) -> dict[str, Any]:
        payload, _ = self._get(self.repo_api(repo, f"/commits/{sha}"))
        return payload

    def fetch_compare(self, repo: str, from_commit: str, to_commit: str) -> dict[str, Any]:
        payload, _ = self._get(self.repo_api(repo, f"/compare/{from_commit}...{to_commit}"))
        return payload


def parse_commit_messages(commits: Iterable[dict[str, Any]]) -> list[str]:
    messages: list[str] = []
    for commit in commits:
        if "message" in commit and commit["message"]:
            messages.append(str(commit["message"]))
        elif isinstance(commit.get("commit"), dict):
            message = commit["commit"].get("message")
            if message:
                messages.append(str(message))
    return messages


def fetch_missing_pr_info_from_github(repo: str, step: PRStep, github: GitHubClient) -> None:
    details: dict[str, Any] | None = None
    commits: list[dict[str, Any]] | None = None

    need_details = not step.title or not step.body or not step.to_commit or not step.from_commit
    if need_details:
        details = github.fetch_pr_details(repo, step.pr_number)
        step.title = step.title or details.get("title") or ""
        step.body = step.body or details.get("body") or ""
        step.to_commit = step.to_commit or details.get("merge_commit_sha") or ""
        if not step.from_commit:
            step.from_commit = (
                details.get("base", {}).get("sha")
                or details.get("base", {}).get("commit", {}).get("sha")
                or ""
            )

    if not step.commit_messages:
        commits = github.fetch_pr_commits(repo, step.pr_number)
        step.commit_messages = parse_commit_messages(commits)

    if not step.from_commit and step.to_commit:
        commit_details = github.fetch_commit(repo, step.to_commit)
        parents = commit_details.get("parents") or []
        if parents:
            step.from_commit = parents[0].get("sha") or ""
        step.merge_commit_message = (
            commit_details.get("commit", {}).get("message") or step.merge_commit_message
        )

    if step.to_commit and not step.merge_commit_message:
        try:
            commit_details = github.fetch_commit(repo, step.to_commit)
            step.merge_commit_message = commit_details.get("commit", {}).get("message") or ""
            if not step.from_commit:
                parents = commit_details.get("parents") or []
                if parents:
                    step.from_commit = parents[0].get("sha") or ""
        except SWEIterError:
            if details and details.get("merge_commit_sha"):
                step.merge_commit_message = ""


def build_requirement_text(step: PRStep) -> str:
    body = (step.body or "").strip()
    if body:
        step.requirement_source = "pull_request"
        step.requirement_description = body
        return body

    title = (step.title or "").strip()
    merge_message = (step.merge_commit_message or "").strip()
    fallback = "\n\n".join(part for part in [title, merge_message] if part)
    if fallback:
        step.requirement_source = "title_and_merge_commit"
        step.requirement_description = fallback
        return fallback

    commit_text = "\n\n".join(message.strip() for message in step.commit_messages if message.strip())
    if commit_text:
        step.requirement_source = "commit_messages"
        step.requirement_description = commit_text
        return commit_text
    return ""


def parse_chain_json(input_path: Path, github: GitHubClient) -> tuple[str, str, list[PRStep]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    repo = data.get("repo")
    if not repo:
        raise SWEIterError("Error: input JSON must contain repo.")
    chain = data.get("chain")
    if not isinstance(chain, list) or not chain:
        raise SWEIterError("Error: input JSON must contain chain.")
    if chain[0].get("type") != "base":
        raise SWEIterError("Error: chain[0].type must be base.")
    base_commit = chain[0].get("sha") or chain[0].get("commit") or ""
    if not base_commit:
        raise SWEIterError("Error: base node must contain sha.")

    steps: list[PRStep] = []
    for node in chain[1:]:
        if node.get("type") != "pr":
            continue
        pr_number = node.get("pr_number")
        if pr_number is None:
            raise SWEIterError("Error: PR node must contain pr_number.")
        step = PRStep(
            step_id=len(steps) + 1,
            pr_number=int(pr_number),
            title=node.get("title") or "",
            body=node.get("body") or "",
            commit_messages=parse_commit_messages(node.get("commits") or []),
            merge_commit_message=node.get("merge_commit_message") or "",
            from_commit=node.get("mainline_parent_sha") or node.get("from_commit") or "",
            to_commit=node.get("merge_commit_sha") or node.get("to_commit") or "",
        )
        if (
            not step.from_commit
            or not step.to_commit
            or not (step.title or step.body or step.commit_messages)
        ):
            fetch_missing_pr_info_from_github(repo, step, github)
        requirement = build_requirement_text(step)
        if not step.from_commit or not step.to_commit:
            raise SWEIterError(
                f"Error: PR #{step.pr_number} must have mainline_parent_sha and merge_commit_sha."
            )
        if not requirement:
            raise SWEIterError(
                f"Error: PR #{step.pr_number} must have title/body/commit messages."
            )
        steps.append(step)

    if not steps:
        raise SWEIterError("Error: chain must contain at least one PR node.")
    return repo, base_commit, steps


def clone_repo(repo: str, cache_root: Path, log_path: Path) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    repo_path = cache_root / repo_slug(repo)
    clone_url = f"https://github.com/{repo}.git"
    if repo_path.exists():
        if not (repo_path / ".git").exists():
            raise SWEIterError(f"Error: cache path exists but is not a git repo: {repo_path}")
        run_command(["git", "fetch", "--all", "--tags", "--prune"], cwd=repo_path, log_path=log_path)
    else:
        run_command(["git", "clone", clone_url, str(repo_path)], log_path=log_path)
    return repo_path


def checkout_commit(repo_path: Path, commit: str, log_path: Path | None = None) -> None:
    run_command(["git", "checkout", "--force", commit], cwd=repo_path, log_path=log_path)


def git_file_exists(repo_path: Path, commit: str, file_path: str) -> bool:
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{file_path}"],
        cwd=repo_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def git_show_file(repo_path: Path, commit: str, file_path: str) -> bytes | None:
    proc = subprocess.run(
        ["git", "show", f"{commit}:{file_path}"],
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    name = Path(normalized).name.lower()
    parts = [part.lower() for part in normalized.split("/")]
    return (
        any(part in {"test", "tests", "spec", "specs"} for part in parts[:-1])
        or name == "conftest.py"
        or (name.startswith("test_") and name.endswith(".py"))
        or name.endswith("_test.py")
        or (name.startswith("spec_") and name.endswith(".py"))
        or name.endswith("_spec.py")
    )


def is_docs_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    name = Path(normalized).name.lower()
    parts = [part.lower() for part in normalized.split("/")]
    return (
        name.startswith("readme")
        or "docs" in parts
        or normalized.lower().endswith(".md")
        or normalized.lower().endswith(".rst")
    )


def is_source_path(path: str) -> bool:
    return path.endswith(".py") and not is_test_path(path) and not is_docs_path(path)


def classify_file_stats(files: list[dict[str, Any]]) -> dict[str, Any]:
    stats = {
        "changed_files": [],
        "source_changed_files": [],
        "test_changed_files": [],
        "docs_changed_files": [],
        "add_lines_total": 0,
        "delete_lines_total": 0,
        "add_lines_src": 0,
        "delete_lines_src": 0,
        "add_lines_tests": 0,
        "delete_lines_tests": 0,
        "add_lines_docs": 0,
        "delete_lines_docs": 0,
    }
    for item in files:
        filename = item.get("filename") or item.get("path") or ""
        if not filename:
            continue
        additions = int(item.get("additions") or 0)
        deletions = int(item.get("deletions") or 0)
        stats["changed_files"].append(filename)
        stats["add_lines_total"] += additions
        stats["delete_lines_total"] += deletions
        if is_test_path(filename):
            stats["test_changed_files"].append(filename)
            stats["add_lines_tests"] += additions
            stats["delete_lines_tests"] += deletions
        elif is_docs_path(filename):
            stats["docs_changed_files"].append(filename)
            stats["add_lines_docs"] += additions
            stats["delete_lines_docs"] += deletions
        elif is_source_path(filename):
            stats["source_changed_files"].append(filename)
            stats["add_lines_src"] += additions
            stats["delete_lines_src"] += deletions
    for key in [
        "changed_files",
        "source_changed_files",
        "test_changed_files",
        "docs_changed_files",
    ]:
        stats[key] = sorted(set(stats[key]))
    return stats


def local_diff_stats(repo_path: Path, from_commit: str, to_commit: str, log_path: Path) -> dict[str, Any]:
    proc = run_command(
        ["git", "diff", "--numstat", from_commit, to_commit],
        cwd=repo_path,
        log_path=log_path,
        check=True,
    )
    files: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        additions = 0 if parts[0] == "-" else int(parts[0])
        deletions = 0 if parts[1] == "-" else int(parts[1])
        path = parts[2]
        files.append({"filename": path, "additions": additions, "deletions": deletions})
    return classify_file_stats(files)


def fetch_compare_stats_from_github(
    repo: str,
    repo_path: Path,
    from_commit: str,
    to_commit: str,
    github: GitHubClient,
    log_path: Path,
) -> dict[str, Any]:
    try:
        compare = github.fetch_compare(repo, from_commit, to_commit)
        files = compare.get("files") or []
        if files:
            return classify_file_stats(files)
    except SWEIterError as exc:
        append_log(log_path, f"GitHub compare failed; falling back to local diff: {exc}")
    return local_diff_stats(repo_path, from_commit, to_commit, log_path)


def extract_golden_patch(repo_path: Path, step: PRStep, output_path: Path, log_path: Path) -> None:
    proc = run_command(
        ["git", "diff", "--binary", step.from_commit, step.to_commit],
        cwd=repo_path,
        log_path=log_path,
        check=True,
    )
    output_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    if not proc.stdout.strip():
        raise SWEIterError(f"Error: GoldenPatch is empty for PR #{step.pr_number}.")
    step.golden_patch_file = str(output_path)


def iter_repo_py_files(repo_path: Path) -> Iterable[Path]:
    ignored = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
    for path in repo_path.rglob("*.py"):
        if any(part in ignored for part in path.relative_to(repo_path).parts):
            continue
        yield path


def scan_test_files(repo_path: Path) -> list[str]:
    tests: set[str] = set()
    for py_file in iter_repo_py_files(repo_path):
        rel = py_file.relative_to(repo_path).as_posix()
        if is_test_path(rel):
            tests.add(rel)
    return sorted(tests)


def ensure_python_repo(repo_path: Path) -> None:
    markers = [
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "pytest.ini",
        "tox.ini",
    ]
    if any((repo_path / marker).exists() for marker in markers):
        return
    if scan_test_files(repo_path):
        return
    raise SWEIterError(
        "Error: target repository is not recognized as a Python repository. "
        "SWE-Iter v1 only supports Python repositories."
    )


def venv_python(repo_path: Path) -> Path:
    return repo_path / ".venv" / "bin" / "python"


def build_venv(repo_path: Path, log_path: Path, label: str = "repo") -> None:
    append_log(log_path, f"\n# Building Python virtual environment for {label}: {repo_path}")
    try:
        existing_venv = repo_path / ".venv"
        if existing_venv.exists():
            shutil.rmtree(existing_venv)
        run_command(["python3", "-m", "venv", ".venv"], cwd=repo_path, log_path=log_path)
        py = str(venv_python(repo_path))
        run_command(
            [py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            cwd=repo_path,
            log_path=log_path,
        )
        if (repo_path / "requirements.txt").exists():
            run_command(
                [py, "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=repo_path,
                log_path=log_path,
            )
        if any((repo_path / marker).exists() for marker in ["pyproject.toml", "setup.py", "setup.cfg"]):
            editable_with_test = run_command(
                [py, "-m", "pip", "install", "-e", ".[test]"],
                cwd=repo_path,
                log_path=log_path,
                check=False,
            )
            if editable_with_test.returncode != 0:
                editable = run_command(
                    [py, "-m", "pip", "install", "-e", "."],
                    cwd=repo_path,
                    log_path=log_path,
                    check=False,
                )
                if editable.returncode != 0:
                    append_log(
                        log_path,
                        "Editable install failed; continuing with repository root on sys.path.",
                    )
        install_poetry_lock_dependencies(repo_path, log_path)
        run_command(
            [py, "-m", "pip", "install", "pytest", "pytest-json-report"],
            cwd=repo_path,
            log_path=log_path,
        )
    except SWEIterError as exc:
        raise SWEIterError("Error: failed to build Python virtual environment for repository.") from exc


def discover_tests(repo_path: Path, log_path: Path) -> list[str]:
    tests = scan_test_files(repo_path)
    if not tests:
        append_log(log_path, "No pytest test files found; continuing with PatchScore-only evaluation.")
        return []
    proc = run_command(
        [str(venv_python(repo_path)), "-m", "pytest", "--collect-only", "-q"],
        cwd=repo_path,
        log_path=log_path,
        check=False,
        timeout=300,
    )
    if proc.returncode != 0:
        missing_modules = extract_missing_modules(f"{proc.stdout}\n{proc.stderr}")
        if missing_modules and install_missing_pytest_dependencies(repo_path, missing_modules, log_path):
            proc = run_command(
                [str(venv_python(repo_path)), "-m", "pytest", "--collect-only", "-q"],
                cwd=repo_path,
                log_path=log_path,
                check=False,
                timeout=300,
            )
        if proc.returncode != 0:
            raise SWEIterError(
                f"Error: pytest collect failed. SWE-Iter v1 requires executable tests. "
                f"See log: {log_path}"
            )
    collected_tests = collect_test_files_from_pytest_output(proc.stdout)
    if not collected_tests:
        raise SWEIterError(
            f"Error: pytest did not collect executable tests. SWE-Iter v1 requires executable tests. "
            f"See log: {log_path}"
        )
    return collected_tests


def collect_test_files_from_pytest_output(output: str) -> list[str]:
    test_files: set[str] = set()
    for node_id in collect_test_cases_from_pytest_output(output):
        file_part = test_file_from_nodeid(node_id)
        if file_part:
            test_files.add(file_part)
    return sorted(test_files)


def collect_test_cases_from_pytest_output(output: str) -> list[str]:
    test_cases: set[str] = set()
    for line in output.splitlines():
        node_id = line.strip()
        if "::" not in node_id:
            continue
        file_part = test_file_from_nodeid(node_id)
        if file_part:
            test_cases.add(node_id)
    return sorted(test_cases)


def test_file_from_nodeid(nodeid: str) -> str:
    file_part = nodeid.split("::", 1)[0]
    return file_part if file_part.endswith(".py") else ""


def normalize_pytest_nodeids(
    repo_path: Path,
    nodeids: Iterable[str],
    test_targets: Iterable[str],
) -> list[str]:
    target_files = sorted(
        {
            test_file_from_nodeid(target) or target
            for target in test_targets
            if (test_file_from_nodeid(target) or target).endswith(".py")
        }
    )
    basename_map: dict[str, list[str]] = {}
    for test_file in target_files:
        basename_map.setdefault(Path(test_file).name, []).append(test_file)

    normalized: set[str] = set()
    for nodeid in nodeids:
        if "::" not in nodeid:
            continue
        file_part, rest = nodeid.split("::", 1)
        if (repo_path / file_part).exists() or file_part in target_files:
            normalized_file = file_part
        else:
            candidates = basename_map.get(Path(file_part).name, [])
            normalized_file = candidates[0] if len(candidates) == 1 else file_part
        normalized.add(f"{Path(normalized_file).as_posix()}::{rest}")
    return sorted(normalized)


def normalize_pytest_result_nodeids(
    repo_path: Path,
    results: dict[str, bool],
    test_targets: Iterable[str],
) -> dict[str, bool]:
    output: dict[str, bool] = {}
    for raw_nodeid, passed in results.items():
        normalized_nodeids = normalize_pytest_nodeids(repo_path, [raw_nodeid], test_targets)
        if normalized_nodeids:
            output[normalized_nodeids[0]] = passed
    return output


def failed_pytest_targets_for_timeout(
    repo_path: Path,
    test_targets: Iterable[str],
) -> dict[str, bool]:
    failed: dict[str, bool] = {}
    for target in test_targets:
        if "::" in target:
            normalized = normalize_pytest_nodeids(repo_path, [target], test_targets)
            failed[normalized[0] if normalized else target] = False
        else:
            failed[target] = False
    return failed


def extract_missing_modules(output: str) -> set[str]:
    modules: set[str] = set()
    patterns = [
        r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]",
        r"ImportError:\s+No module named ['\"]([^'\"]+)['\"]",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, output):
            modules.add(match.split(".")[0])
    return modules


def pyproject_dependency_names(repo_path: Path) -> set[str]:
    pyproject = repo_path / "pyproject.toml"
    if not pyproject.exists():
        return set()
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return set()

    names: set[str] = set()

    def add_mapping_keys(mapping: Any) -> None:
        if isinstance(mapping, dict):
            for name in mapping:
                if str(name).lower() != "python":
                    names.add(str(name))

    poetry = data.get("tool", {}).get("poetry", {})
    add_mapping_keys(poetry.get("dev-dependencies"))
    groups = poetry.get("group", {})
    if isinstance(groups, dict):
        for group_name in ["test", "tests", "testing", "dev"]:
            add_mapping_keys(groups.get(group_name, {}).get("dependencies"))

    dependency_groups = data.get("dependency-groups", {})
    if isinstance(dependency_groups, dict):
        for group_name in ["test", "tests", "testing", "dev"]:
            values = dependency_groups.get(group_name, [])
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str):
                        names.add(re.split(r"[<>=!~\[\]; ]", value, maxsplit=1)[0])

    return {name for name in names if name}


def is_likely_test_dependency(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    return normalized.startswith("pytest") or normalized in {
        "attrs",
        "coverage",
        "freezegun",
        "hypothesis",
        "mock",
        "pytest-cov",
        "requests-mock",
        "responses",
    }


def poetry_declared_lock_dependency_names(repo_path: Path) -> set[str]:
    pyproject = repo_path / "pyproject.toml"
    if not pyproject.exists():
        return set()
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return set()
    poetry = data.get("tool", {}).get("poetry", {})
    if not poetry:
        return set()

    names: set[str] = set()

    def add_runtime_dependencies(mapping: Any) -> None:
        if isinstance(mapping, dict):
            for name, spec in mapping.items():
                if str(name).lower() != "python":
                    if isinstance(spec, dict) and spec.get("optional") is True:
                        continue
                    names.add(str(name))

    def add_test_dependencies(mapping: Any) -> None:
        if isinstance(mapping, dict):
            for name in mapping:
                if is_likely_test_dependency(str(name)):
                    names.add(str(name))

    add_runtime_dependencies(poetry.get("dependencies"))
    add_test_dependencies(poetry.get("dev-dependencies"))
    groups = poetry.get("group", {})
    if isinstance(groups, dict):
        for group_name in ["test", "tests", "testing", "dev"]:
            add_test_dependencies(groups.get(group_name, {}).get("dependencies"))
    return names


def poetry_lock_pinned_requirements(repo_path: Path) -> list[str]:
    lock_path = repo_path / "poetry.lock"
    if not lock_path.exists():
        return []
    names = {
        name.lower().replace("_", "-")
        for name in poetry_declared_lock_dependency_names(repo_path)
    }
    if not names:
        return []
    try:
        lock_data = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return []
    requirements: list[str] = []
    for package in lock_data.get("package", []):
        if not isinstance(package, dict):
            continue
        name = str(package.get("name") or "")
        version = str(package.get("version") or "")
        if name.lower().replace("_", "-") in names and version:
            requirements.append(f"{name}=={version}")
    return sorted(set(requirements), key=str.lower)


def install_poetry_lock_dependencies(repo_path: Path, log_path: Path) -> bool:
    requirements = poetry_lock_pinned_requirements(repo_path)
    if not requirements:
        return False
    append_log(log_path, f"Installing Poetry lock pinned dependencies: {requirements}")
    run_command(
        [str(venv_python(repo_path)), "-m", "pip", "install", *requirements],
        cwd=repo_path,
        log_path=log_path,
    )
    return True


def dependency_candidates_for_module(module_name: str, available: set[str]) -> list[str]:
    aliases = {
        "attr": "attrs",
        "attrs": "attrs",
        "PIL": "Pillow",
        "yaml": "PyYAML",
    }
    normalized_available = {name.lower().replace("_", "-"): name for name in available}
    candidates: list[str] = []
    for candidate in [aliases.get(module_name, module_name), module_name]:
        normalized = candidate.lower().replace("_", "-")
        if normalized in normalized_available:
            candidates.append(normalized_available[normalized])
    return sorted(set(candidates))


def install_missing_pytest_dependencies(
    repo_path: Path,
    missing_modules: set[str],
    log_path: Path,
) -> bool:
    available = pyproject_dependency_names(repo_path)
    to_install: list[str] = []
    for module in sorted(missing_modules):
        to_install.extend(dependency_candidates_for_module(module, available))
    to_install = sorted(set(to_install))
    if not to_install:
        return False
    append_log(
        log_path,
        "Detected missing pytest import modules "
        f"{sorted(missing_modules)}; installing matching pyproject test/dev dependencies {to_install}.",
    )
    run_command(
        [str(venv_python(repo_path)), "-m", "pip", "install", *to_install],
        cwd=repo_path,
        log_path=log_path,
    )
    return True


def pytest_environment_error(output: str) -> bool:
    lower = output.lower()
    return any(
        marker in lower
        for marker in [
            "error collecting",
            "importerror while importing test module",
            "modulenotfounderror",
            "no module named",
            "internalerror",
        ]
    )


def restore_overlay(restore_data: dict[Path, bytes | None]) -> None:
    for path, original in restore_data.items():
        if original is None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            parent = path.parent
            while parent.exists() and not any(parent.iterdir()):
                if parent.name in {".", ""}:
                    break
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(original)
        clear_pycache_for(path)


def clear_pycache_for(path: Path) -> None:
    pycache = path.parent / "__pycache__"
    if not pycache.exists():
        return
    for cache_file in pycache.glob(f"{path.stem}.*.pyc"):
        try:
            cache_file.unlink()
        except FileNotFoundError:
            pass


def parse_pytest_json_report(report_path: Path) -> dict[str, bool]:
    if not report_path.exists():
        return {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SWEIterError(f"Error: pytest JSON report is not valid JSON: {report_path}") from exc

    results: dict[str, bool] = {}
    for item in report.get("tests") or []:
        if not isinstance(item, dict):
            continue
        nodeid = item.get("nodeid")
        if not isinstance(nodeid, str) or "::" not in nodeid:
            continue
        results[nodeid] = item.get("outcome") == "passed"
    return results


def collect_pytest_cases(
    repo_path: Path,
    test_files: list[str],
    log_path: Path,
    overlay_sources: dict[str, bytes] | None = None,
    overlay_mode: str = "none",
    timeout: int = 300,
) -> list[str]:
    if not test_files:
        return []
    restore_data: dict[Path, bytes | None] = {}
    overlay_sources = overlay_sources or {}

    for rel_path, content in overlay_sources.items():
        target = repo_path / rel_path
        should_overlay = overlay_mode == "all" or (overlay_mode == "missing" and not target.exists())
        if not should_overlay:
            continue
        if target not in restore_data:
            restore_data[target] = target.read_bytes() if target.exists() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        clear_pycache_for(target)

    try:
        existing = [test_file for test_file in test_files if (repo_path / test_file).exists()]
        if not existing:
            return []
        proc = run_command(
            [str(venv_python(repo_path)), "-m", "pytest", "--collect-only", "-q", *existing],
            cwd=repo_path,
            log_path=log_path,
            check=False,
            timeout=timeout,
        )
        combined = f"{proc.stdout}\n{proc.stderr}"
        if proc.returncode != 0:
            if pytest_environment_error(combined):
                raise SWEIterError("Error: pytest environment/import error during collection.")
            raise SWEIterError("Error: pytest collect failed for selected test files.")
        return normalize_pytest_nodeids(
            repo_path,
            collect_test_cases_from_pytest_output(proc.stdout),
            existing,
        )
    finally:
        restore_overlay(restore_data)


def run_pytest_files(
    repo_path: Path,
    test_targets: list[str],
    log_path: Path,
    overlay_sources: dict[str, bytes] | None = None,
    overlay_mode: str = "none",
    timeout_per_file: int = 60,
) -> dict[str, bool]:
    if not test_targets:
        return {}
    restore_data: dict[Path, bytes | None] = {}
    overlay_sources = overlay_sources or {}

    for rel_path, content in overlay_sources.items():
        target = repo_path / rel_path
        should_overlay = overlay_mode == "all" or (overlay_mode == "missing" and not target.exists())
        if not should_overlay:
            continue
        if target not in restore_data:
            restore_data[target] = target.read_bytes() if target.exists() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        clear_pycache_for(target)

    results: dict[str, bool] = {}
    try:
        target_files: set[str] = set()
        for test_target in test_targets:
            file_part = test_file_from_nodeid(test_target) or test_target
            if not file_part or not (repo_path / file_part).exists():
                raise SWEIterError(f"Error: pytest test target does not exist: {test_target}")
            target_files.add(file_part)

        report_path = log_path.with_name(f"{log_path.stem}_pytest_report.json")
        timeout = max(timeout_per_file, min(timeout_per_file * max(1, len(target_files)), 3600))
        try:
            proc = run_command(
                [
                    str(venv_python(repo_path)),
                    "-m",
                    "pytest",
                    *test_targets,
                    "-q",
                    "--json-report",
                    f"--json-report-file={report_path}",
                ],
                cwd=repo_path,
                log_path=log_path,
                check=False,
                timeout=timeout,
            )
        except SWEIterError as exc:
            if "command timed out" not in str(exc):
                raise
            append_log(
                log_path,
                f"Pytest timed out after {timeout}s; marking selected targets failed.",
            )
            results.update(failed_pytest_targets_for_timeout(repo_path, test_targets))
            return results
        combined = f"{proc.stdout}\n{proc.stderr}"
        if proc.returncode in {0, 1}:
            if pytest_environment_error(combined):
                raise SWEIterError("Error: pytest environment/import error.")
            parsed = parse_pytest_json_report(report_path)
            if not parsed:
                raise SWEIterError("Error: pytest produced no case-level results.")
            results.update(normalize_pytest_result_nodeids(repo_path, parsed, test_targets))
        else:
            raise SWEIterError("Error: pytest crashed.")
    finally:
        restore_overlay(restore_data)
    return results


def classify_f2p_p2p_tests(
    repo_path: Path,
    step: PRStep,
    all_test_files: list[str],
    logs_dir: Path,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    def empty_classification(reason: str) -> tuple[dict[str, Any], dict[str, bytes]]:
        return {
            "selected_test_files": [],
            "selected_test_cases": [],
            "from_results": {},
            "to_results": {},
            "F2P": [],
            "P2P": [],
            "P2F": [],
            "F2F": [],
            "test_selection_reason": reason,
        }, {}

    if not all_test_files:
        return empty_classification("no_pytest_test_files")

    changed_test_files = step.patch_stats.get("test_changed_files") or []
    selected = sorted({test_file for test_file in changed_test_files if is_test_path(test_file)})
    if not selected:
        return empty_classification("no_changed_test_files")

    selected_existing = [
        test_file for test_file in selected if git_file_exists(repo_path, step.to_commit, test_file)
    ]
    if not selected_existing:
        return empty_classification("no_changed_test_files_exist_at_to_commit")

    test_sources: dict[str, bytes] = {}
    for test_file in selected_existing:
        content = git_show_file(repo_path, step.to_commit, test_file)
        if content is not None:
            test_sources[test_file] = content

    to_log = logs_dir / f"step_{step.step_id:03d}_to.log"
    from_log = logs_dir / f"step_{step.step_id:03d}_from.log"

    checkout_commit(repo_path, step.to_commit, to_log)
    to_cases = collect_pytest_cases(repo_path, selected_existing, to_log)

    checkout_commit(repo_path, step.from_commit, from_log)
    from_existing = [
        test_file for test_file in selected_existing if git_file_exists(repo_path, step.from_commit, test_file)
    ]
    from_cases = collect_pytest_cases(repo_path, from_existing, from_log)
    selected_cases = sorted(set(to_cases) - set(from_cases))
    if not selected_cases:
        return empty_classification("no_new_test_cases")

    checkout_commit(repo_path, step.to_commit, to_log)
    to_results = run_pytest_files(repo_path, selected_cases, to_log)

    checkout_commit(repo_path, step.from_commit, from_log)
    from_results = run_pytest_files(
        repo_path,
        selected_cases,
        from_log,
        overlay_sources=test_sources,
        overlay_mode="all",
    )

    buckets = {"F2P": [], "P2P": [], "P2F": [], "F2F": []}
    for test_case in selected_cases:
        from_pass = from_results[test_case]
        to_pass = to_results[test_case]
        if not from_pass and to_pass:
            buckets["F2P"].append(test_case)
        elif from_pass and to_pass:
            buckets["P2P"].append(test_case)
        elif from_pass and not to_pass:
            buckets["P2F"].append(test_case)
        else:
            buckets["F2F"].append(test_case)

    return {
        "selected_test_files": selected_existing,
        "selected_test_cases": selected_cases,
        "from_results": from_results,
        "to_results": to_results,
        "F2P": buckets["F2P"],
        "P2P": buckets["P2P"],
        "P2F": buckets["P2F"],
        "F2F": buckets["F2F"],
        "test_selection_reason": "new_test_cases",
    }, test_sources


def create_model_worktree(
    repo_path: Path,
    worktree_root: Path,
    slug: str,
    source_repo_url: str,
    final_commit: str,
    base_commit: str,
    log_path: Path,
) -> Path:
    worktree_root.mkdir(parents=True, exist_ok=True)
    model_path = worktree_root / f"{slug}__model"
    if model_path.exists():
        shutil.rmtree(model_path)
    run_command(
        ["git", "clone", "--no-checkout", str(repo_path), str(model_path)],
        cwd=repo_path,
        log_path=log_path,
    )
    checkout_commit(model_path, base_commit, log_path)
    shutil.rmtree(model_path / ".git")
    run_command(["git", "init"], cwd=model_path, log_path=log_path)
    run_command(
        ["git", "add", "--all", "--", "."],
        cwd=model_path,
        log_path=log_path,
    )
    run_command(
        [
            "git",
            "-c",
            "user.name=SWE-Iter Evaluator",
            "-c",
            "user.email=swe-iter@example.invalid",
            "commit",
            "-m",
            "SWE-Iter base",
        ],
        cwd=model_path,
        log_path=log_path,
    )
    configure_local_fetch_remote(model_path, log_path)
    return model_path


def configure_local_fetch_remote(repo_path: Path, log_path: Path | None = None) -> None:
    """Point origin at the copied repo itself so SWE-agent reset avoids network fetches."""
    remotes = run_command(["git", "remote"], cwd=repo_path, log_path=log_path, check=False)
    if "origin" in remotes.stdout.splitlines():
        run_command(
            ["git", "remote", "set-url", "origin", "."],
            cwd=repo_path,
            log_path=log_path,
            check=False,
        )
    else:
        run_command(
            ["git", "remote", "add", "origin", "."],
            cwd=repo_path,
            log_path=log_path,
            check=False,
        )


def load_resume_step_results(
    result_path: Path,
    repo: str,
    base_commit: str,
    final_commit: str,
    steps: list[PRStep],
) -> list[dict[str, Any]]:
    if not result_path.exists():
        return []
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if (
        data.get("repo") != repo
        or data.get("base_commit") != base_commit
        or data.get("final_commit") != final_commit
    ):
        return []
    completed: list[dict[str, Any]] = []
    for expected, saved in zip(steps, data.get("steps", [])):
        if (
            saved.get("step_id") != expected.step_id
            or saved.get("pr_number") != expected.pr_number
            or not saved.get("scores")
        ):
            break
        completed.append(saved)
    return completed


def hydrate_step_from_result(step: PRStep, saved: dict[str, Any]) -> None:
    step.golden_patch_file = str(saved.get("golden_patch_file") or step.golden_patch_file)
    step.agent_result = dict(saved.get("agent_result") or {})
    step.atomic_requirements = list(saved.get("atomic_requirements") or [])
    step.patch_judgment = dict(saved.get("patch_judgment") or {})
    step.scores = dict(saved.get("scores") or {})


def remove_runtime_artifacts(repo_path: Path) -> None:
    for name in [".venv", ".pytest_cache", ".mypy_cache", "trajectories"]:
        path = repo_path / name
        if path.exists():
            shutil.rmtree(path)
    for path in repo_path.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path)


def collect_worktree_diff(repo_path: Path, log_path: Path | None = None) -> str:
    status = run_command(
        ["git", "status", "--porcelain"],
        cwd=repo_path,
        log_path=log_path,
        check=True,
    )

    def ignore_untracked(path: str) -> bool:
        normalized = path.replace("\\", "/")
        parts = normalized.split("/")
        return (
            normalized.startswith(".venv/")
            or normalized.startswith(".pytest_cache/")
            or normalized.startswith(".mypy_cache/")
            or normalized.startswith("trajectories/")
            or "__pycache__" in parts
            or normalized.endswith(".pyc")
        )

    untracked = [
        line[3:]
        for line in status.stdout.splitlines()
        if line.startswith("?? ") and not ignore_untracked(line[3:])
    ]
    if untracked:
        run_command(["git", "add", "-N", "--", *untracked], cwd=repo_path, log_path=log_path)
    diff = run_command(
        [
            "git",
            "diff",
            "--binary",
            "--",
            ".",
            ":(exclude).venv/**",
            ":(exclude).pytest_cache/**",
            ":(exclude).mypy_cache/**",
            ":(exclude)trajectories/**",
            ":(exclude)**/__pycache__/**",
            ":(exclude)**/*.pyc",
        ],
        cwd=repo_path,
        log_path=log_path,
        check=True,
    )
    return diff.stdout


def collect_model_cumulative_diff(repo_path: Path, log_path: Path | None = None) -> str:
    status = run_command(
        ["git", "status", "--porcelain"],
        cwd=repo_path,
        log_path=log_path,
        check=True,
    )

    def ignore_untracked(path: str) -> bool:
        normalized = path.replace("\\", "/")
        parts = normalized.split("/")
        return (
            normalized.startswith(".venv/")
            or normalized.startswith(".pytest_cache/")
            or normalized.startswith(".mypy_cache/")
            or normalized.startswith("trajectories/")
            or "__pycache__" in parts
            or normalized.endswith(".pyc")
        )

    untracked = [
        line[3:]
        for line in status.stdout.splitlines()
        if line.startswith("?? ") and not ignore_untracked(line[3:])
    ]
    if untracked:
        run_command(["git", "add", "-N", "--", *untracked], cwd=repo_path, log_path=log_path)
    root_commit = run_command(
        ["git", "rev-list", "--max-parents=0", "HEAD"],
        cwd=repo_path,
        log_path=log_path,
        check=True,
    ).stdout.splitlines()[0]
    diff = run_command(
        [
            "git",
            "diff",
            "--binary",
            root_commit,
            "--",
            ".",
            ":(exclude).venv/**",
            ":(exclude).pytest_cache/**",
            ":(exclude).mypy_cache/**",
            ":(exclude)trajectories/**",
            ":(exclude)**/__pycache__/**",
            ":(exclude)**/*.pyc",
        ],
        cwd=repo_path,
        log_path=log_path,
        check=True,
    )
    return diff.stdout


def submission_patches_from_trajectories(repo_path: Path) -> list[str]:
    patches: list[str] = []
    trajectories_dir = repo_path / "trajectories"
    if not trajectories_dir.exists():
        return patches
    for traj_path in sorted(trajectories_dir.rglob("*.traj"), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            data = json.loads(traj_path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            continue
        submission = data.get("info", {}).get("submission")
        if isinstance(submission, str) and submission.strip().startswith("diff --git "):
            patches.append(submission)
    return patches


def write_noop_model_patch(paths: RuntimePaths, step: PRStep, reason: str) -> dict[str, Any]:
    output_patch = paths.patches_model / f"step_{step.step_id:03d}_pr_{step.pr_number}_model.patch"
    output_patch.write_text("", encoding="utf-8")
    return {
        "problem_statement_file": None,
        "stdout_log": None,
        "stderr_log": None,
        "model_patch_file": str(output_patch),
        "returncode": 0,
        "skipped_swe_agent": True,
        "skip_reason": reason,
    }


def should_skip_swe_agent_for_step(step: PRStep) -> bool:
    return not bool(step.patch_stats.get("source_changed_files"))


def non_source_requirement_type(step: PRStep) -> str | None:
    if not should_skip_swe_agent_for_step(step):
        return None
    if step.patch_stats.get("test_changed_files"):
        return "test"
    if step.patch_stats.get("docs_changed_files"):
        return "docs"
    return "non_source"


def coerce_no_source_requirement_types(step: PRStep) -> None:
    req_type = non_source_requirement_type(step)
    if req_type is None:
        return
    for requirement in step.atomic_requirements:
        original_type = requirement.get("type")
        if original_type and str(original_type).strip().lower() != req_type:
            requirement.setdefault("original_type", original_type)
        requirement["type"] = req_type


def commit_model_step(repo_path: Path, step: PRStep, log_path: Path) -> str | None:
    remove_runtime_artifacts(repo_path)
    run_command(
        [
            "git",
            "add",
            "--all",
            "--",
            ".",
            ":(exclude).venv/**",
            ":(exclude).pytest_cache/**",
            ":(exclude).mypy_cache/**",
            ":(exclude)trajectories/**",
            ":(exclude)**/__pycache__/**",
            ":(exclude)**/*.pyc",
        ],
        cwd=repo_path,
        log_path=log_path,
        check=True,
    )
    diff_check = run_command(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo_path,
        log_path=log_path,
        check=False,
    )
    if diff_check.returncode == 0:
        return None
    if diff_check.returncode != 1:
        raise SWEIterError(f"Error: failed to inspect staged model changes for step {step.step_id}.")
    run_command(
        [
            "git",
            "-c",
            "user.name=SWE-Iter Evaluator",
            "-c",
            "user.email=swe-iter@example.invalid",
            "commit",
            "-m",
            f"SWE-Iter model step {step.step_id:03d} PR #{step.pr_number}",
        ],
        cwd=repo_path,
        log_path=log_path,
        check=True,
    )
    sha = run_command(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        log_path=log_path,
        check=True,
    ).stdout.strip()
    return sha or None


def truncate_text(text: str, max_chars: int = 20000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


def read_text_safe(path: str | Path, max_chars: int | None = None) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if max_chars:
        return truncate_text(text, max_chars=max_chars)
    return text


def write_swe_agent_problem_statement(
    step: PRStep,
    previous_requirements: list[str],
    output_path: Path,
) -> None:
    previous = "\n".join(
        f"{index}. {truncate_text(req.strip(), 500)}"
        for index, req in enumerate(previous_requirements, start=1)
    )
    if not previous:
        previous = "None yet."

    content = f"""# SWE-Iter Step {step.step_id} / PR #{step.pr_number}

You are working on a real repository-level iterative development task.

This repository is being developed through a sequence of merged pull requests.
You are currently at step {step.step_id}. Previous steps have already been
implemented in the current working tree. Your task is to implement the new
requirement below while preserving all previously implemented behavior.

## Current requirement

Title: {step.title}

Description:

{step.requirement_description}

## Previous requirements that must remain satisfied

{previous}

## Rules

- Do not modify tests unless absolutely necessary. For this benchmark, treat tests as hidden validation.
- Do not inspect any GoldenPatch or oracle patch files.
- Do not inspect patches_gold/, eval_results.json, summary.md, or any scoring artifacts.
- Do not inspect any oracle files outside the repository worktree.
- Make minimal, maintainable changes.
- Preserve backward compatibility with prior steps.
- This step continues from the existing model code. Do not restart from scratch.
- The goal is to satisfy this requirement while keeping all earlier requirements from regressing.
"""
    output_path.write_text(content, encoding="utf-8")


class SWEAgentRunner:
    def __init__(self, config: dict[str, Any], paths: RuntimePaths) -> None:
        self.config = config
        self.paths = paths

    def build_command(self, model_worktree: Path, problem_statement: Path) -> list[str]:
        swe_config = self.config.get("swe_agent", {})
        command = swe_config.get("command") or "sweagent"
        command_parts = shlex.split(command)
        template = swe_config.get("command_template") or []
        config_path = swe_config.get("config_path") or ""
        extra_args = list(swe_config.get("extra_args") or [])

        values = {
            "command": command,
            "config_path": config_path,
            "model_worktree_path": str(model_worktree),
            "problem_statement_path": str(problem_statement),
        }
        if template:
            rendered: list[str] = []
            for part in template:
                rendered.append(str(part).format(**values))
            return rendered + extra_args

        cmd = command_parts + ["run"]
        if config_path and not str(config_path).startswith("FILL_"):
            cmd += ["--config", str(config_path)]
        cmd += [
            "--env.repo.path",
            str(model_worktree),
            "--env.repo.type",
            "local",
            "--problem_statement.path",
            str(problem_statement),
        ]
        cmd += extra_args
        return cmd

    def find_patch_file_candidates(self, stdout: str, stderr: str, model_worktree: Path) -> list[Path]:
        text = f"{stdout}\n{stderr}"
        candidates: list[Path] = []
        for match in re.findall(r"(?P<path>(?:/|\.{1,2}/|[\w.-]+/)[^\s:'\"]+\.(?:patch|diff))", text):
            candidate = Path(match)
            if not candidate.is_absolute():
                candidate = (model_worktree / candidate).resolve()
            if candidate.exists() and candidate.is_file():
                candidates.append(candidate)
        return candidates

    def collect_swe_agent_patch_or_diff(
        self,
        model_worktree: Path,
        step: PRStep,
        stdout: str,
        stderr: str,
        log_path: Path,
    ) -> Path:
        output_patch = self.paths.patches_model / f"step_{step.step_id:03d}_pr_{step.pr_number}_model.patch"
        diff = collect_worktree_diff(model_worktree, log_path=log_path)
        if diff.strip():
            output_patch.write_text(diff, encoding="utf-8", errors="replace")
            remove_runtime_artifacts(model_worktree)
            return output_patch

        candidates = self.find_patch_file_candidates(stdout, stderr, model_worktree)
        trajectory_patches = submission_patches_from_trajectories(model_worktree)
        if candidates:
            patch_file = candidates[0]
            shutil.copyfile(patch_file, output_patch)
        elif trajectory_patches:
            output_patch.write_text(trajectory_patches[0], encoding="utf-8", errors="replace")
        else:
            remove_runtime_artifacts(model_worktree)
            output_patch.write_text("", encoding="utf-8")
            return output_patch
        apply_locally = self.config.get("swe_agent", {}).get("apply_patch_locally", True)
        if apply_locally:
            run_command(["git", "apply", str(output_patch)], cwd=model_worktree, log_path=log_path)
            diff = collect_worktree_diff(model_worktree, log_path=log_path)
            if diff.strip():
                output_patch.write_text(diff, encoding="utf-8", errors="replace")
        remove_runtime_artifacts(model_worktree)
        if not output_patch.read_text(encoding="utf-8", errors="replace").strip():
            raise SWEIterError("Error: SWE-agent patch is empty.")
        return output_patch

    def run_swe_agent_for_step(
        self,
        model_worktree: Path,
        step: PRStep,
        previous_requirements: list[str],
    ) -> dict[str, Any]:
        problem_statement = (
            self.paths.problem_statements / f"step_{step.step_id:03d}_pr_{step.pr_number}.md"
        )
        write_swe_agent_problem_statement(step, previous_requirements, problem_statement)
        remove_runtime_artifacts(model_worktree)
        stdout_log = self.paths.logs / f"step_{step.step_id:03d}_sweagent_stdout.log"
        stderr_log = self.paths.logs / f"step_{step.step_id:03d}_sweagent_stderr.log"
        combined_log = self.paths.logs / f"step_{step.step_id:03d}_sweagent_command.log"
        cmd = self.build_command(model_worktree, problem_statement)
        timeout = int(self.config.get("swe_agent", {}).get("timeout_seconds_per_step") or 3600)
        proc = run_command(
            cmd,
            cwd=model_worktree,
            log_path=combined_log,
            check=False,
            timeout=timeout,
        )
        stdout_log.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
        stderr_log.write_text(proc.stderr or "", encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise SWEIterError(
                f"Error: SWE-agent failed for step {step.step_id}. "
                f"See logs: {stdout_log}, {stderr_log}"
            )
        patch_path = self.collect_swe_agent_patch_or_diff(
            model_worktree, step, proc.stdout or "", proc.stderr or "", combined_log
        )
        return {
            "problem_statement_file": str(problem_statement),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "model_patch_file": str(patch_path),
            "returncode": proc.returncode,
            "no_patch_produced": not patch_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip(),
        }


class DeepSeekClient:
    def __init__(self, config: dict[str, Any], log_path: Path | None = None) -> None:
        api_config = config.get("api", {}).get("deepseek_pro", {})
        self.api_key = api_config.get("api_key") or ""
        self.base_url = api_config.get("base_url") or ""
        self.model = api_config.get("model") or "DeepSeek-V4-Pro"
        self.log_path = log_path
        if not self.api_key or str(self.api_key).startswith("FILL_"):
            raise SWEIterError("Error: DeepSeek Pro API key is required for PatchScore.")
        if not self.base_url or str(self.base_url).startswith("FILL_"):
            raise SWEIterError("Error: DeepSeek Pro base_url is required for PatchScore.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SWEIterError("Error: openai package is required for DeepSeek Pro calls.") from exc
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def json_chat(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if self.log_path:
            append_log(self.log_path, f"DeepSeek call model={self.model}")
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                return json.loads(content)
            except Exception as exc:  # noqa: BLE001 - external API failures are normalized.
                last_exc = exc
                if self.log_path:
                    append_log(self.log_path, f"DeepSeek attempt {attempt} failed: {exc}")
                time.sleep(2 * attempt)
        raise SWEIterError("Error: DeepSeek Pro API call failed.") from last_exc


def golden_patch_summary(step: PRStep) -> str:
    patch_text = read_text_safe(step.golden_patch_file, max_chars=18000)
    changed = ", ".join(step.patch_stats.get("changed_files") or [])
    return f"Changed files: {changed}\n\nGoldenPatch excerpt:\n{patch_text}"


def extract_atomic_requirements_with_deepseek_pro(
    client: DeepSeekClient,
    step: PRStep,
) -> list[dict[str, Any]]:
    system_prompt = (
        "You extract atomic software requirements from PR information. "
        "Return only valid JSON."
    )
    user_prompt = f"""
Extract atomic requirements for this PR step.

Return JSON exactly shaped like:
{{
  "atomic_requirements": [
    {{
      "id": "R1",
      "requirement": "...",
      "type": "feature|bug_fix|compatibility|edge_case|refactor|test|docs|other",
      "evidence": ["PR body", "commit message", "golden patch"],
      "evidence_strength": 0.8,
      "must_have": true
    }}
  ]
}}

PR title:
{step.title}

PR body:
{step.body}

Commit messages:
{json.dumps(step.commit_messages, ensure_ascii=False, indent=2)}

Patch stats:
{json.dumps(step.patch_stats, ensure_ascii=False, indent=2)}

{golden_patch_summary(step)}
"""
    payload = client.json_chat(system_prompt, user_prompt)
    requirements = payload.get("atomic_requirements")
    if not isinstance(requirements, list):
        raise SWEIterError("Error: PatchScore atomic requirement output is not valid JSON.")
    normalized: list[dict[str, Any]] = []
    for index, req in enumerate(requirements, start=1):
        if not isinstance(req, dict):
            continue
        req.setdefault("id", f"R{index}")
        req.setdefault("type", "other")
        req.setdefault("evidence", [])
        req.setdefault("evidence_strength", 1.0)
        req.setdefault("must_have", True)
        if req.get("requirement"):
            normalized.append(req)
    if not normalized:
        raise SWEIterError("Error: no atomic requirements extracted.")
    return normalized


def namespace_atomic_requirements(step: PRStep, requirements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    namespaced: list[dict[str, Any]] = []
    for req in requirements:
        item = dict(req)
        local_id = str(item.get("id") or f"R{len(namespaced) + 1}")
        item["local_id"] = local_id
        item["id"] = f"S{step.step_id}:{local_id}"
        item["step_id"] = step.step_id
        item["pr_number"] = step.pr_number
        namespaced.append(item)
    return namespaced


def judge_patch_with_deepseek_pro(
    client: DeepSeekClient,
    cumulative_requirements: list[dict[str, Any]],
    step: PRStep,
    model_patch_text: str,
    test_result_summary: dict[str, Any],
) -> dict[str, Any]:
    system_prompt = (
        "You judge semantic satisfaction of software requirements by a model patch. "
        "Use the GoldenPatch only as reference, not for textual similarity. "
        "Return only valid JSON."
    )
    user_prompt = f"""
Judge whether the current cumulative model patch satisfies the cumulative atomic requirements.
For requirements whose type is "test" or "docs", still provide a judgment, but
remember that SWE-agent is instructed not to modify tests or documentation unless
absolutely necessary. The evaluator will keep those judgments as evidence context
and will exclude test/docs requirements from code PatchScore aggregation.

Return JSON exactly shaped like:
{{
  "requirement_judgments": [
    {{
      "id": "S1:R1",
      "satisfied": true,
      "confidence": 0.9,
      "reason": "..."
    }}
  ],
  "patch_score": 0.75
}}

Cumulative atomic requirements:
{json.dumps(cumulative_requirements, ensure_ascii=False, indent=2)}

Use each requirement's exact unique id in your judgments. Requirement ids are
namespaced by step, such as "S1:R1" and "S2:R1"; never collapse them by local id.

Current step:
PR #{step.pr_number}
Title: {step.title}
Description:
{step.requirement_description}

Changed files:
{json.dumps(step.patch_stats.get("changed_files", []), ensure_ascii=False)}

Test results summary:
{json.dumps(test_result_summary, ensure_ascii=False, indent=2)}

GoldenPatch reference summary:
{golden_patch_summary(step)}

Model cumulative patch from the chain base to the current model state:
{truncate_text(model_patch_text, max_chars=24000)}
"""
    payload = client.json_chat(system_prompt, user_prompt)
    judgments = payload.get("requirement_judgments")
    if not isinstance(judgments, list):
        raise SWEIterError("Error: PatchScore judgment output is not valid JSON.")
    return payload


def compute_test_score(
    model_test_results: dict[str, bool],
    cumulative_f2p: set[str],
    cumulative_p2p: set[str],
    f2p_weight: float = 0.7,
    p2p_weight: float = 0.3,
) -> tuple[float, dict[str, Any]]:
    if not cumulative_f2p and not cumulative_p2p:
        raise SWEIterError("Error: no F2P or P2P tests available for scoring.")

    def pass_rate(tests: set[str]) -> float | None:
        if not tests:
            return None
        return sum(1 for test in tests if model_test_results.get(test, False)) / len(tests)

    f2p_rate = pass_rate(cumulative_f2p)
    p2p_rate = pass_rate(cumulative_p2p)
    if f2p_rate is None:
        score = float(p2p_rate)
    elif p2p_rate is None:
        score = float(f2p_rate)
    else:
        score = f2p_weight * f2p_rate + p2p_weight * p2p_rate
    return score, {
        "f2p_pass_rate": f2p_rate,
        "p2p_pass_rate": p2p_rate,
        "cumulative_f2p_count": len(cumulative_f2p),
        "cumulative_p2p_count": len(cumulative_p2p),
    }


def compute_patch_score(
    cumulative_requirements: list[dict[str, Any]],
    judgment: dict[str, Any],
) -> float:
    score, _ = compute_patch_score_details(cumulative_requirements, judgment)
    return score


def requirement_type(req: dict[str, Any]) -> str:
    return str(req.get("type") or "other").strip().lower()


def is_patch_score_requirement(req: dict[str, Any]) -> bool:
    if req.get("must_have") is False:
        return False
    return requirement_type(req) not in PATCH_SCORE_EXCLUDED_REQUIREMENT_TYPES


def compute_patch_score_details(
    cumulative_requirements: list[dict[str, Any]],
    judgment: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    judgments = {
        str(item.get("id")): item
        for item in judgment.get("requirement_judgments", [])
        if isinstance(item, dict)
    }
    if not cumulative_requirements:
        raise SWEIterError("Error: no atomic requirements available for PatchScore.")
    included_requirements = [
        req for req in cumulative_requirements if is_patch_score_requirement(req)
    ]
    excluded_requirements = [
        req for req in cumulative_requirements if not is_patch_score_requirement(req)
    ]
    details = {
        "patch_score_included_requirement_ids": [
            str(req.get("id")) for req in included_requirements
        ],
        "patch_score_excluded_requirement_ids": [
            str(req.get("id")) for req in excluded_requirements
        ],
        "patch_score_excluded_requirement_types": sorted(
            {
                requirement_type(req)
                for req in excluded_requirements
                if requirement_type(req) in PATCH_SCORE_EXCLUDED_REQUIREMENT_TYPES
            }
        ),
        "patch_score_no_code_requirements": False,
    }
    if not included_requirements:
        details["patch_score_no_code_requirements"] = True
        details["patch_score_weighted_numerator"] = 0.0
        details["patch_score_weighted_denominator"] = 0.0
        return 1.0, details

    numerator = 0.0
    denominator = 0.0
    for req in included_requirements:
        req_id = str(req.get("id"))
        weight = float(req.get("evidence_strength") or 1.0)
        denominator += weight
        if judgments.get(req_id, {}).get("satisfied") is True:
            numerator += weight
    if denominator == 0:
        raise SWEIterError("Error: PatchScore denominator is zero.")
    details["patch_score_weighted_numerator"] = numerator
    details["patch_score_weighted_denominator"] = denominator
    return numerator / denominator, details


def compute_evidence(
    cumulative_tests: set[str],
    cumulative_requirements: list[dict[str, Any]],
    rho: float,
    kappa_tests: int,
) -> tuple[float, dict[str, Any]]:
    test_evidence = min(1.0, len(cumulative_tests) / max(kappa_tests, 1))
    if cumulative_requirements:
        req_evidence = sum(
            float(req.get("evidence_strength") or 1.0) for req in cumulative_requirements
        ) / len(cumulative_requirements)
    else:
        req_evidence = 0.0
    evidence = rho * test_evidence + (1 - rho) * req_evidence
    return evidence, {"test_evidence": test_evidence, "req_evidence": req_evidence}


def compute_iter_score(step_results: list[dict[str, Any]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for step in step_results:
        scores = step.get("scores", {})
        weight = float(scores.get("weight") or 0.0)
        step_score = float(scores.get("StepScore") or 0.0)
        numerator += weight * step_score
        denominator += weight
    if denominator <= 0:
        raise SWEIterError("Error: IterScore denominator is zero.")
    return numerator / denominator


def run_model_cumulative_tests(
    model_worktree: Path,
    cumulative_tests: set[str],
    test_sources: dict[str, bytes],
    log_path: Path,
) -> dict[str, bool]:
    selected = sorted(cumulative_tests)
    if not selected:
        append_log(log_path, "No cumulative tests selected; skipping pytest run.")
        return {}
    selected_files = {test_file_from_nodeid(test) for test in selected}
    overlay = {test: test_sources[test] for test in selected_files if test in test_sources}
    return run_pytest_files(
        model_worktree,
        selected,
        log_path,
        overlay_sources=overlay,
        overlay_mode="all",
    )


def write_results_json(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def summary_test_mode(result: dict[str, Any], scores: dict[str, Any]) -> str:
    if not result.get("all_test_files"):
        return "PatchScore-only (no pytest test files found)"
    if scores.get("test_score_unavailable"):
        reason = scores.get("test_score_unavailable_reason")
        if reason in {
            "no_new_test_cases",
            "no_changed_test_files",
            "no_changed_test_files_exist_at_to_commit",
        }:
            return "PatchScore-only (no new pytest test cases for this step)"
        if reason == "no_scored_reference_test_cases":
            return "PatchScore-only (new tests did not yield scored F2P/P2P cases)"
        return "PatchScore-only"
    return "pytest + PatchScore"


def write_summary_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        f"# SWE-Iter Summary: {result.get('repo')}",
        "",
        f"- Repo: `{result.get('repo')}`",
        f"- Chain length: {result.get('chain_length')}",
        f"- Base commit: `{result.get('base_commit')}`",
        f"- Final commit: `{result.get('final_commit')}`",
        f"- IterScore: `{result.get('IterScore')}`",
        f"- Test files discovered: `{len(result.get('all_test_files') or [])}`",
        "- TestScore granularity: `pytest case/nodeid`",
    ]
    if not result.get("all_test_files"):
        lines.append("- Evaluation mode: `PatchScore-only (no pytest test files found)`")
        lines.append("- Test status: `No pytest test files found; TestScore mirrors PatchScore for reporting.`")
    lines.extend(["", "## Steps", ""])
    for step in result.get("steps", []):
        scores = step.get("scores", {})
        tests = step.get("tests", {})
        requirement = step.get("requirement", {})
        lines.extend(
            [
                f"### Step {step.get('step_id')} / PR #{step.get('pr_number')}",
                "",
                f"- Title: {requirement.get('title', '')}",
                f"- Requirement source: {requirement.get('source', '')}",
                f"- F2P/P2P cases: {len(tests.get('F2P', []))}/{len(tests.get('P2P', []))}",
                f"- Test mode: `{summary_test_mode(result, scores)}`",
                f"- TestScore: `{scores.get('TestScore')}`",
                f"- PatchScore: `{scores.get('PatchScore')}`",
                f"- StepScore: `{scores.get('StepScore')}`",
                f"- Confidence: `{scores.get('Confidence')}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def print_startup_banner(repo: str, config: dict[str, Any]) -> None:
    swe_config = config.get("swe_agent", {})
    deepseek = config.get("api", {}).get("deepseek_pro", {})
    print(f"Repo: {repo}")
    print(f"SWE-agent command: {swe_config.get('command')}")
    print(f"SWE-agent config_path: {swe_config.get('config_path')}")
    print(f"DeepSeek Pro model: {deepseek.get('model')}")


def prepare_steps(
    repo: str,
    repo_path: Path,
    steps: list[PRStep],
    all_test_files: list[str],
    github: GitHubClient,
    paths: RuntimePaths,
) -> None:
    for step in steps:
        patch_path = paths.patches_gold / f"step_{step.step_id:03d}_pr_{step.pr_number}.patch"
        extract_golden_patch(repo_path, step, patch_path, paths.logs / "golden_patches.log")
        step.patch_stats = fetch_compare_stats_from_github(
            repo,
            repo_path,
            step.from_commit,
            step.to_commit,
            github,
            paths.logs / "api.log",
        )
        tests, sources = classify_f2p_p2p_tests(repo_path, step, all_test_files, paths.logs)
        step.tests = tests
        step.test_sources = sources


def run_evaluation(
    input_path: Path,
    output_dir: Path,
    config_path: Path | None,
    cache_dir: Path,
    worktree_dir: Path,
) -> dict[str, Any]:
    paths = ensure_dirs(output_dir)
    config = load_config(config_path)
    github = GitHubClient(
        config.get("api", {}).get("github", {}).get("token"),
        log_path=paths.logs / "api.log",
    )
    repo, base_commit, steps = parse_chain_json(input_path, github)
    print_startup_banner(repo, config)

    clone_log = paths.logs / "clone.log"
    repo_path = clone_repo(repo, cache_dir, clone_log)
    final_commit = steps[-1].to_commit

    env_log = paths.logs / "env_setup.log"
    checkout_commit(repo_path, final_commit, paths.logs / "checkout.log")
    ensure_python_repo(repo_path)
    build_venv(repo_path, env_log, label="evaluation repo")
    all_test_files = discover_tests(repo_path, paths.logs / "pytest_collect.log")

    prepare_steps(repo, repo_path, steps, all_test_files, github, paths)

    resume_step_results = load_resume_step_results(
        paths.output_dir / "eval_results.json",
        repo,
        base_commit,
        final_commit,
        steps,
    )
    resume_count = len(resume_step_results)
    model_worktree = worktree_dir / f"{repo_slug(repo)}__model"
    if resume_count:
        if not (model_worktree / ".git").exists():
            raise SWEIterError(
                "Error: cannot resume because model worktree is missing: "
                f"{model_worktree}"
            )
        configure_local_fetch_remote(model_worktree, env_log)
        append_log(env_log, f"Resuming from {resume_count} completed steps.")
    else:
        model_worktree = create_model_worktree(
            repo_path,
            worktree_dir,
            repo_slug(repo),
            f"https://github.com/{repo}.git",
            final_commit,
            base_commit,
            env_log,
        )

    deepseek = DeepSeekClient(config, log_path=paths.logs / "deepseek.log")
    runner = SWEAgentRunner(config, paths)

    cumulative_f2p: set[str] = set()
    cumulative_p2p: set[str] = set()
    cumulative_test_sources: dict[str, bytes] = {}
    cumulative_requirements: list[dict[str, Any]] = []
    previous_requirements: list[str] = []

    result: dict[str, Any] = {
        "repo": repo,
        "base_commit": base_commit,
        "final_commit": final_commit,
        "chain_length": len(steps),
        "all_test_files": all_test_files,
        "steps": [],
        "IterScore": None,
    }

    for step, saved in zip(steps, resume_step_results):
        hydrate_step_from_result(step, saved)
    if resume_count:
        result["steps"] = [step.as_result() for step in steps[:resume_count]]

    for step in steps:
        if step.step_id <= resume_count:
            previous_requirements.append(step.requirement_description)
            cumulative_test_sources.update(step.test_sources)
            cumulative_f2p.update(step.tests.get("F2P", []))
            cumulative_p2p.update(step.tests.get("P2P", []))
            cumulative_requirements.extend(step.atomic_requirements)
            continue

        step.atomic_requirements = namespace_atomic_requirements(
            step,
            extract_atomic_requirements_with_deepseek_pro(deepseek, step),
        )
        coerce_no_source_requirement_types(step)
        cumulative_requirements.extend(step.atomic_requirements)

        if should_skip_swe_agent_for_step(step):
            step.agent_result = write_noop_model_patch(
                paths,
                step,
                "step has no source changes in the golden patch",
            )
        else:
            step.agent_result = runner.run_swe_agent_for_step(
                model_worktree,
                step,
                previous_requirements,
            )
        previous_requirements.append(step.requirement_description)
        build_venv(model_worktree, env_log, label="model worktree")

        cumulative_test_sources.update(step.test_sources)
        cumulative_f2p.update(step.tests.get("F2P", []))
        cumulative_p2p.update(step.tests.get("P2P", []))
        cumulative_tests = cumulative_f2p | cumulative_p2p
        model_test_log = paths.logs / f"step_{step.step_id:03d}_model_tests.log"
        has_new_test_cases = bool(step.tests.get("selected_test_cases"))
        has_scored_tests = bool(cumulative_tests)
        if has_new_test_cases and has_scored_tests:
            model_test_results = run_model_cumulative_tests(
                model_worktree,
                cumulative_tests,
                cumulative_test_sources,
                model_test_log,
            )
            test_score, test_score_details = compute_test_score(
                model_test_results,
                cumulative_f2p,
                cumulative_p2p,
                f2p_weight=float(config["scoring"]["test_f2p_weight"]),
                p2p_weight=float(config["scoring"]["test_p2p_weight"]),
            )
        else:
            if has_new_test_cases:
                reason = "no_scored_reference_test_cases"
            elif not all_test_files:
                reason = "no_pytest_test_files"
            else:
                reason = str(step.tests.get("test_selection_reason") or "no_new_test_cases")
            model_test_results = {}
            test_score = None
            test_score_details = {
                "f2p_pass_rate": None,
                "p2p_pass_rate": None,
                "cumulative_f2p_count": len(cumulative_f2p),
                "cumulative_p2p_count": len(cumulative_p2p),
                "test_score_unavailable": True,
                "test_score_unavailable_reason": reason,
            }

        model_cumulative_patch = collect_model_cumulative_diff(
            model_worktree,
            paths.logs / f"step_{step.step_id:03d}_model_cumulative_patch.log",
        )
        model_cumulative_patch_file = (
            paths.patches_model / f"step_{step.step_id:03d}_pr_{step.pr_number}_model_cumulative.patch"
        )
        model_cumulative_patch_file.write_text(
            model_cumulative_patch,
            encoding="utf-8",
            errors="replace",
        )
        step.agent_result["model_cumulative_patch_file"] = str(model_cumulative_patch_file)
        model_patch_text = truncate_text(model_cumulative_patch, max_chars=30000)
        test_summary = {
            "model_test_results": model_test_results,
            **test_score_details,
        }
        step.patch_judgment = judge_patch_with_deepseek_pro(
            deepseek,
            cumulative_requirements,
            step,
            model_patch_text,
            test_summary,
        )
        patch_score, patch_score_details = compute_patch_score_details(
            cumulative_requirements,
            step.patch_judgment,
        )

        lambda_test = float(config["scoring"]["lambda_test"])
        if test_score is None:
            test_score = patch_score
            effective_lambda_test = 0.0
        else:
            effective_lambda_test = lambda_test
        rho = float(config["scoring"]["rho_evidence"])
        gamma = float(config["scoring"]["gamma_time"])
        kappa_tests = int(config["scoring"].get("kappa_tests") or 20)

        step_score = effective_lambda_test * test_score + (1 - effective_lambda_test) * patch_score
        agree = 1 - abs(test_score - patch_score)
        evidence, evidence_details = compute_evidence(
            cumulative_tests,
            cumulative_requirements,
            rho,
            kappa_tests,
        )
        confidence = agree * evidence
        delta_src = int(step.patch_stats.get("add_lines_src") or 0) + int(
            step.patch_stats.get("delete_lines_src") or 0
        )
        weight = (gamma**step.step_id) * math.log(1 + delta_src) * confidence

        step.scores = {
            "TestScore": test_score,
            "PatchScore": patch_score,
            "StepScore": step_score,
            "Agree": agree,
            "Evidence": evidence,
            "Confidence": confidence,
            "effective_lambda_test": effective_lambda_test,
            "weight": weight,
            "delta_src": delta_src,
            **test_score_details,
            **patch_score_details,
            **evidence_details,
        }
        model_commit_log = paths.logs / f"step_{step.step_id:03d}_model_commit.log"
        model_commit_sha = commit_model_step(model_worktree, step, model_commit_log)
        step.agent_result["model_commit"] = model_commit_sha
        result["steps"] = [prepared.as_result() for prepared in steps[: step.step_id]]
        write_results_json(paths.output_dir / "eval_results.json", result)

    result["steps"] = [step.as_result() for step in steps]
    result["IterScore"] = compute_iter_score(result["steps"])
    write_results_json(paths.output_dir / "eval_results.json", result)
    write_summary_md(paths.output_dir / "summary.md", result)
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SWE-Iter v1 evaluation.")
    parser.add_argument("--input", required=True, help="Path to PR chain JSON.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument(
        "--cache-dir",
        default=".cache/repos",
        help="Repository clone cache directory.",
    )
    parser.add_argument(
        "--worktree-dir",
        default=".cache/worktrees",
        help="Model worktree cache directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        run_evaluation(
            input_path=Path(args.input).resolve(),
            output_dir=Path(args.output).resolve(),
            config_path=Path(args.config).resolve() if args.config else None,
            cache_dir=Path(args.cache_dir).resolve(),
            worktree_dir=Path(args.worktree_dir).resolve(),
        )
    except SWEIterError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
