import json
import subprocess
import sys

import pytest

from swe_iter_eval import (
    PRStep,
    RuntimePaths,
    SWEAgentRunner,
    SWEIterError,
    build_requirement_text,
    classify_f2p_p2p_tests,
    classify_file_stats,
    coerce_no_source_requirement_types,
    collect_model_cumulative_diff,
    collect_test_files_from_pytest_output,
    commit_model_step,
    compute_patch_score,
    compute_patch_score_details,
    compute_test_score,
    dependency_candidates_for_module,
    create_model_worktree,
    discover_tests,
    extract_missing_modules,
    is_docs_path,
    is_source_path,
    is_test_path,
    namespace_atomic_requirements,
    poetry_lock_pinned_requirements,
    pyproject_dependency_names,
    run_model_cumulative_tests,
    run_pytest_files,
    should_skip_swe_agent_for_step,
    submission_patches_from_trajectories,
    write_summary_md,
)


def test_path_classification():
    assert is_test_path("tests/test_widget.py")
    assert is_test_path("pkg/widget_test.py")
    assert is_test_path("pkg/spec_widget.py")
    assert is_test_path("pkg/widget_spec.py")
    assert is_test_path("conftest.py")
    assert not is_test_path("rich/_inspect.py")
    assert not is_test_path("src/contest.py")
    assert is_docs_path("README.md")
    assert is_docs_path("docs/usage.rst")
    assert is_source_path("src/widget.py")
    assert not is_source_path("tests/test_widget.py")


def test_classify_file_stats_splits_lines():
    stats = classify_file_stats(
        [
            {"filename": "src/widget.py", "additions": 10, "deletions": 2},
            {"filename": "tests/test_widget.py", "additions": 5, "deletions": 1},
            {"filename": "README.md", "additions": 3, "deletions": 0},
        ]
    )
    assert stats["changed_files"] == ["README.md", "src/widget.py", "tests/test_widget.py"]
    assert stats["source_changed_files"] == ["src/widget.py"]
    assert stats["test_changed_files"] == ["tests/test_widget.py"]
    assert stats["docs_changed_files"] == ["README.md"]
    assert stats["add_lines_src"] == 10
    assert stats["delete_lines_tests"] == 1
    assert stats["add_lines_docs"] == 3


def test_should_skip_swe_agent_when_golden_patch_has_no_source_changes():
    step = PRStep(
        step_id=1,
        pr_number=5,
        patch_stats=classify_file_stats(
            [
                {"filename": "tests/test_widget.py", "additions": 5, "deletions": 1},
                {"filename": "README.md", "additions": 3, "deletions": 0},
            ]
        ),
    )
    assert should_skip_swe_agent_for_step(step)


def test_should_not_skip_swe_agent_when_golden_patch_has_source_changes():
    step = PRStep(
        step_id=1,
        pr_number=5,
        patch_stats=classify_file_stats(
            [
                {"filename": "src/widget.py", "additions": 1, "deletions": 1},
                {"filename": "tests/test_widget.py", "additions": 5, "deletions": 1},
            ]
        ),
    )
    assert not should_skip_swe_agent_for_step(step)


def test_collect_test_files_from_pytest_output_excludes_helpers():
    output = """
tests/test_card.py::test_card_render
tests/test_tree.py::test_render_ascii
852 tests collected in 0.17s
"""
    assert collect_test_files_from_pytest_output(output) == [
        "tests/test_card.py",
        "tests/test_tree.py",
    ]


def test_discover_tests_returns_empty_for_repo_without_tests(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'sample'\nversion = '0.1.0'\n", encoding="utf-8")
    assert discover_tests(tmp_path, tmp_path / "collect.log") == []


def test_run_pytest_files_returns_empty_when_no_tests_selected(tmp_path):
    assert run_pytest_files(tmp_path, [], tmp_path / "pytest.log") == {}


def test_run_model_cumulative_tests_returns_empty_when_no_tests_selected(tmp_path):
    assert run_model_cumulative_tests(tmp_path, set(), {}, tmp_path / "model_tests.log") == {}


def test_model_cumulative_tests_use_only_incremental_test_sources(tmp_path):
    python_bin = tmp_path / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text(f"#!/bin/sh\nexec {json.dumps(sys.executable)} \"$@\"\n", encoding="utf-8")
    python_bin.chmod(0o755)

    test_file = "tests/test_future.py"
    step1 = PRStep(
        step_id=1,
        pr_number=1,
        test_sources={test_file: b"def test_current_step():\n    assert True\n"},
    )
    step2 = PRStep(
        step_id=2,
        pr_number=2,
        test_sources={test_file: b"def test_future_step():\n    assert False\n"},
    )

    cumulative_sources: dict[str, bytes] = {}
    cumulative_sources.update(step1.test_sources)
    assert run_model_cumulative_tests(
        tmp_path,
        {test_file},
        cumulative_sources,
        tmp_path / "step1.log",
    ) == {test_file: True}

    cumulative_sources.update(step2.test_sources)
    assert run_model_cumulative_tests(
        tmp_path,
        {test_file},
        cumulative_sources,
        tmp_path / "step2.log",
    ) == {test_file: False}


def test_classify_f2p_p2p_tests_returns_empty_when_no_tests_available(tmp_path):
    step = PRStep(step_id=1, pr_number=5)
    tests, sources = classify_f2p_p2p_tests(tmp_path, step, [], tmp_path)
    assert tests == {
        "selected_test_files": [],
        "from_results": {},
        "to_results": {},
        "F2P": [],
        "P2P": [],
        "P2F": [],
        "F2F": [],
    }
    assert sources == {}


def test_summary_marks_patchscore_only_when_no_test_files(tmp_path):
    summary_path = tmp_path / "summary.md"
    write_summary_md(
        summary_path,
        {
            "repo": "example/repo",
            "chain_length": 1,
            "base_commit": "abc",
            "final_commit": "def",
            "IterScore": 1.0,
            "all_test_files": [],
            "steps": [
                {
                    "step_id": 1,
                    "pr_number": 5,
                    "requirement": {"title": "Fix bug", "source": "pull_request"},
                    "tests": {"F2P": [], "P2P": []},
                    "scores": {
                        "TestScore": 1.0,
                        "PatchScore": 1.0,
                        "StepScore": 1.0,
                        "Confidence": 0.5,
                        "test_score_unavailable": True,
                    },
                }
            ],
        },
    )
    text = summary_path.read_text(encoding="utf-8")
    assert "Evaluation mode: `PatchScore-only (no pytest test files found)`" in text
    assert "Test mode: `PatchScore-only (no pytest test files found)`" in text
    assert "No pytest test files found" in text


def test_requirement_text_prefers_body():
    step = PRStep(
        step_id=1,
        pr_number=5,
        title="Title",
        body="Body requirement",
        commit_messages=["Commit requirement"],
    )
    assert build_requirement_text(step) == "Body requirement"
    assert step.requirement_source == "pull_request"


def test_requirement_text_falls_back_to_commits():
    step = PRStep(step_id=1, pr_number=5, commit_messages=["Implement a thing"])
    assert build_requirement_text(step) == "Implement a thing"
    assert step.requirement_source == "commit_messages"


def test_compute_test_score_handles_empty_f2p():
    score, details = compute_test_score(
        {"test_a.py": True, "test_b.py": False},
        cumulative_f2p=set(),
        cumulative_p2p={"test_a.py", "test_b.py"},
    )
    assert score == 0.5
    assert details["p2p_pass_rate"] == 0.5
    assert details["f2p_pass_rate"] is None


def test_compute_test_score_requires_tests():
    with pytest.raises(SWEIterError):
        compute_test_score({}, set(), set())


def test_compute_patch_score_uses_evidence_strength():
    requirements = [
        {"id": "R1", "evidence_strength": 0.75},
        {"id": "R2", "evidence_strength": 0.25},
    ]
    judgment = {
        "requirement_judgments": [
            {"id": "R1", "satisfied": True},
            {"id": "R2", "satisfied": False},
        ]
    }
    assert compute_patch_score(requirements, judgment) == 0.75


def test_compute_patch_score_excludes_test_and_docs_requirements():
    requirements = [
        {"id": "R1", "type": "bug_fix", "evidence_strength": 0.9},
        {"id": "R2", "type": "test", "evidence_strength": 0.8},
        {"id": "R3", "type": "docs", "evidence_strength": 0.7},
    ]
    judgment = {
        "requirement_judgments": [
            {"id": "R1", "satisfied": True},
            {"id": "R2", "satisfied": False},
            {"id": "R3", "satisfied": False},
        ]
    }
    score, details = compute_patch_score_details(requirements, judgment)
    assert score == 1.0
    assert details["patch_score_included_requirement_ids"] == ["R1"]
    assert details["patch_score_excluded_requirement_ids"] == ["R2", "R3"]


def test_compute_patch_score_is_neutral_when_only_test_docs_requirements():
    requirements = [
        {"id": "R1", "type": "test", "evidence_strength": 0.8},
        {"id": "R2", "type": "docs", "evidence_strength": 0.7},
    ]
    score, details = compute_patch_score_details(requirements, {"requirement_judgments": []})
    assert score == 1.0
    assert details["patch_score_no_code_requirements"] is True


def test_no_source_step_requirements_are_excluded_from_patch_score():
    step = PRStep(
        step_id=2,
        pr_number=3324,
        patch_stats=classify_file_stats(
            [
                {"filename": "tests/test_traceback.py", "additions": 0, "deletions": 10},
            ]
        ),
        atomic_requirements=[
            {"id": "S2:R1", "type": "bug_fix", "evidence_strength": 0.9},
            {"id": "S2:R2", "evidence_strength": 0.5},
        ],
    )
    coerce_no_source_requirement_types(step)

    score, details = compute_patch_score_details(
        step.atomic_requirements,
        {"requirement_judgments": [{"id": "S2:R1", "satisfied": False}]},
    )
    assert score == 1.0
    assert [req["type"] for req in step.atomic_requirements] == ["test", "test"]
    assert step.atomic_requirements[0]["original_type"] == "bug_fix"
    assert details["patch_score_no_code_requirements"] is True
    assert details["patch_score_excluded_requirement_ids"] == ["S2:R1", "S2:R2"]


def test_namespace_atomic_requirements_uses_step_scoped_ids():
    step = PRStep(step_id=3, pr_number=3378)
    requirements = namespace_atomic_requirements(
        step,
        [
            {"id": "R1", "requirement": "first"},
            {"id": "R2", "requirement": "second"},
        ],
    )
    assert [req["id"] for req in requirements] == ["S3:R1", "S3:R2"]
    assert [req["local_id"] for req in requirements] == ["R1", "R2"]
    assert [req["pr_number"] for req in requirements] == [3378, 3378]


def test_extract_missing_modules():
    output = "ModuleNotFoundError: No module named 'attr'"
    assert extract_missing_modules(output) == {"attr"}


def test_pyproject_dev_dependency_mapping(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.poetry.dev-dependencies]
attrs = "^21.4.0"
pytest = "^7.0.0"
""",
        encoding="utf-8",
    )
    available = pyproject_dependency_names(tmp_path)
    assert "attrs" in available
    assert dependency_candidates_for_module("attr", available) == ["attrs"]


def test_poetry_lock_pinned_requirements(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.poetry.dependencies]
python = ">=3.10"
pygments = "^2.13.0"
ipywidgets = { version = ">=7.5.1,<9", optional = true }

[tool.poetry.dev-dependencies]
attrs = "^21.4.0"
black = "^22.6"
pytest = "^7.0.0"
""",
        encoding="utf-8",
    )
    (tmp_path / "poetry.lock").write_text(
        """
[[package]]
name = "attrs"
version = "21.4.0"

[[package]]
name = "black"
version = "22.6.0"

[[package]]
name = "pygments"
version = "2.16.1"

[[package]]
name = "ipywidgets"
version = "8.1.1"

[[package]]
name = "pytest"
version = "7.4.2"
""",
        encoding="utf-8",
    )
    assert poetry_lock_pinned_requirements(tmp_path) == [
        "attrs==21.4.0",
        "pygments==2.16.1",
        "pytest==7.4.2",
    ]


def test_commit_model_step_commits_changes(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    (tmp_path / "pkg.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "pkg.py"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
    )

    (tmp_path / "pkg.py").write_text("value = 2\n", encoding="utf-8")
    step = PRStep(step_id=1, pr_number=123)
    commit_sha = commit_model_step(tmp_path, step, tmp_path.parent / "commit.log")

    assert commit_sha
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=tmp_path,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    assert status == ""


def test_collect_model_cumulative_diff_includes_committed_and_worktree_changes(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    (tmp_path / "pkg.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "pkg.py"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "base",
        ],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
    )

    (tmp_path / "pkg.py").write_text("value = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "pkg.py"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "step 1",
        ],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
    )
    (tmp_path / "extra.py").write_text("added = True\n", encoding="utf-8")

    diff = collect_model_cumulative_diff(tmp_path, tmp_path / "diff.log")
    assert "-value = 1" in diff
    assert "+value = 2" in diff
    assert "diff --git a/extra.py b/extra.py" in diff


def test_create_model_worktree_hides_future_commits(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init"], cwd=source, check=True, stdout=subprocess.PIPE)
    (source / "pkg.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "pkg.py"], cwd=source, check=True, stdout=subprocess.PIPE)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "base",
        ],
        cwd=source,
        check=True,
        stdout=subprocess.PIPE,
    )
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    (source / "pkg.py").write_text("value = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "pkg.py"], cwd=source, check=True, stdout=subprocess.PIPE)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "future",
        ],
        cwd=source,
        check=True,
        stdout=subprocess.PIPE,
    )
    future_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()

    model_path = create_model_worktree(
        source,
        tmp_path / "worktrees",
        "sample",
        "https://example.invalid/sample.git",
        future_sha,
        base_sha,
        tmp_path / "create.log",
    )

    assert (model_path / "pkg.py").read_text(encoding="utf-8") == "value = 1\n"
    count = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=model_path,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    assert count == "1"
    future_check = subprocess.run(
        ["git", "cat-file", "-e", f"{future_sha}^{{commit}}"],
        cwd=model_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert future_check.returncode != 0


def test_submission_patches_from_trajectories_reads_autosubmission(tmp_path):
    traj_dir = tmp_path / "trajectories" / "run" / "case"
    traj_dir.mkdir(parents=True)
    patch = (
        "diff --git a/pkg.py b/pkg.py\n"
        "index 257cc56..5716ca5 100644\n"
        "--- a/pkg.py\n"
        "+++ b/pkg.py\n"
        "@@ -1 +1 @@\n"
        "-value = 1\n"
        "+value = 2\n"
    )
    (traj_dir / "case.traj").write_text(
        json.dumps({"info": {"submission": patch}}),
        encoding="utf-8",
    )

    assert submission_patches_from_trajectories(tmp_path) == [patch]


def test_swe_agent_empty_submission_writes_empty_patch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
    (repo / "pkg.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "pkg.py"], cwd=repo, check=True, stdout=subprocess.PIPE)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "base",
        ],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    )
    paths = RuntimePaths(
        output_dir=tmp_path / "out",
        patches_gold=tmp_path / "gold",
        patches_model=tmp_path / "model",
        problem_statements=tmp_path / "problems",
        logs=tmp_path / "logs",
    )
    for path in [
        paths.output_dir,
        paths.patches_gold,
        paths.patches_model,
        paths.problem_statements,
        paths.logs,
    ]:
        path.mkdir()

    runner = SWEAgentRunner({"swe_agent": {"apply_patch_locally": True}}, paths)
    patch_path = runner.collect_swe_agent_patch_or_diff(
        repo,
        PRStep(step_id=12, pr_number=3469),
        stdout="No patch to save.",
        stderr="",
        log_path=tmp_path / "collect.log",
    )

    assert patch_path.read_text(encoding="utf-8") == ""
