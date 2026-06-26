# SWE-Iter Evaluator

This repository contains a v1 SWE-Iter evaluator for iterative software
engineering benchmark runs over mined GitHub PR chains.

The evaluator supports Python repositories with pytest tests. It treats each
merged PR in the input chain as one requirement iteration, calls an external
SWE-agent CLI to generate code, runs file-level F2P/P2P tests, calls a
DeepSeek V4 Pro compatible API for semantic PatchScore, and writes an
IterScore report.

## Configure

Copy the example config:

```bash
cp config.example.yaml config.yaml
```

Then fill:

- `api.github.token`
- `api.deepseek_pro.api_key`
- `api.deepseek_pro.base_url`
- `api.deepseek_pro.model`
- `swe_agent.command`
- `swe_agent.config_path`
- `swe_agent.extra_args`

SWE-agent's code-generation model API key is not stored in this evaluator's
`config.yaml`. Put it in SWE-agent's own config file or in the provider
environment variables required by SWE-agent, such as `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, or `DEEPSEEK_API_KEY`.

You may override selected values with environment variables:

- `GITHUB_TOKEN`
- `DEEPSEEK_V4_PRO_API_KEY`
- `DEEPSEEK_V4_PRO_BASE_URL`
- `SWE_AGENT_CONFIG_PATH`
- `SWE_AGENT_COMMAND`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install and configure SWE-agent separately. The command below should work in
the same shell before you start an evaluation:

```bash
sweagent --help
```

## Run

```bash
python swe_iter_eval.py \
  --input pr_chain_owner_repo.json \
  --config config.yaml \
  --output results/demo
```

At startup the evaluator prints the target repo, SWE-agent command,
SWE-agent config path, and DeepSeek model name. It never prints API keys.

## Input

The input JSON must include:

- `repo`
- `chain`
- `chain[0].type == "base"`
- a base `sha`
- PR nodes with `pr_number`
- enough PR data to resolve `mainline_parent_sha`, `merge_commit_sha`, and a
  natural-language requirement source

If required PR details are missing, the evaluator uses the GitHub REST API to
fetch PR details, PR commits, merge commit metadata, and compare stats.

## Output

The output directory contains:

- `eval_results.json`
- `summary.md`
- `patches_gold/`
- `patches_model/`
- `problem_statements/`
- `logs/`

`eval_results.json` includes the repo metadata, global test files, per-step
requirements, GoldenPatch paths, F2P/P2P/P2F/F2F classifications, model patch
paths, atomic requirements, step scores, and final IterScore.

PatchScore aggregates only non-test, non-docs `must_have` atomic requirements.
Atomic requirements with `type: test` or `type: docs` are retained in the result
for audit and evidence context, but they do not penalize a code-generation agent
that was instructed not to modify tests or documentation.

## Current Limits

- v1 only supports Python repositories.
- v1 requires pytest-discoverable tests.
- v1 exits on environment setup failure.
- v1 uses file-level F2P/P2P classification.
- v1 uses SWE-agent for code generation and DeepSeek V4 Pro for semantic patch
  scoring.
- v1 does not do patch-only fallback when tests, SWE-agent, or DeepSeek fail.

## SWE-agent CLI Variants

By default the evaluator calls:

```bash
sweagent run \
  --config <config_path> \
  --env.repo.path <model_worktree_path> \
  --env.repo.type local \
  --problem_statement.path <problem_statement_path>
```

If your SWE-agent version uses different arguments, set
`swe_agent.command_template` in `config.yaml`. Available placeholders are:

- `{command}`
- `{config_path}`
- `{model_worktree_path}`
- `{problem_statement_path}`

Example:

```yaml
swe_agent:
  command: "sweagent"
  command_template:
    - "{command}"
    - "run"
    - "--config"
    - "{config_path}"
    - "--env.repo.path"
    - "{model_worktree_path}"
    - "--problem_statement.path"
    - "{problem_statement_path}"
```
