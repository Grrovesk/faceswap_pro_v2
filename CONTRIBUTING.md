# Contributing to faceswap_pro v2

We welcome PRs that fix bugs, add features, improve documentation,
or harden the safety policy. Please read this guide before opening
one.

## Before you start

1. **Read [USAGE_POLICY.md](USAGE_POLICY.md).** PRs that weaken
   safety controls, remove watermarks, or facilitate prohibited
   uses will not be merged.
2. **Open an issue first** for anything more than a small fix.
   We'd rather agree on the approach than have you waste effort on
   a PR that doesn't fit.
3. **Search closed issues + PRs.** The thing you want may already
   be done, declined, or in flight.

## Development setup

Follow [INSTALL.md](INSTALL.md) to get a working venv. Then add the
dev extras:

```bash
pip install -r requirements-dev.txt
```

Run the smoke tests:

```bash
pytest tests/
```

## Code style

- **Python 3.10** features only (no walrus-in-comprehension etc that
  shipped in 3.11)
- **4-space indent**, **80-column soft limit** for new lines, no
  trailing whitespace
- **Type hints** on every public function. We don't enforce them
  on private (`_underscore`) helpers but they're appreciated
- **Docstrings** on every module + public function. Module docstrings
  explain the *why*, function docstrings explain inputs/outputs
- **No bare `except`** in production code (orchestrator catches are
  fine because they explicitly log and fall through). Catch
  `Exception` if you really need a wide net and log it
- **Logging**: use `logging.getLogger(__name__)`. Render pipelines
  log through the `log` callable injected from `orchestrator.render()`,
  not via the module logger directly, so log lines surface in the
  Gradio UI

## Architectural principles

These are the load-bearing decisions in the codebase. If your PR
violates one, please motivate it in the description.

1. **Subprocess isolation for upstream repos.** LatentSync and
   RVC each have conflicting dep pins (omegaconf, accelerate,
   diffusers versions). They run in subprocesses, NOT as Python
   imports. Don't bring their requirements into `requirements.txt`.
2. **Typed configs.** Renders take a `LipsyncJob` or `VideoSwapJob`
   dataclass. Don't add kwargs to `orchestrator.render()`; add fields
   to the dataclass.
3. **Snapshots before edits.** When modifying a multi-hundred-line
   file (ui.py, orchestrator.py, pipeline.py, etc.), copy it to
   `_snapshots/<timestamp>_<purpose>/` first. This is enforced by
   convention, not by hooks. If something breaks, the user can
   revert.
4. **No new model downloads without an `_ensure_X()` probe.**
   Auto-download functions live in `core/sam2_install.py`,
   `core/lipsync.py`, `core/gfpgan.py` etc. They check existence,
   download once, cache forever. Don't bake URLs into the main code
   path.
5. **Cancel respects users.** Long-running stages (DDIM inference,
   Demucs separation) check the `cancel_event` at boundaries. If
   you add a new long stage, add a `_check_cancel()` call before
   it.

## Pull request workflow

1. **Fork** the repo and create a feature branch:
   ```
   git checkout -b feat/short-descriptive-name
   ```
2. **Make minimal, surgical commits.** One logical change per
   commit. Reformatting + behavior change in one commit is hard to
   review.
3. **Test.** Run `pytest`. If your change touches the UI, screenshot
   the before/after.
4. **Update docs.** README.md, USAGE.md, CHANGELOG.md as appropriate.
5. **Open the PR** against `main`. Include:
   - What problem does this solve? Link the issue.
   - How does this PR solve it? Architecture summary.
   - What did you test?
   - Did you update docs?

## Reviewing AI-authored PRs

This repo accepts contributions from AI agents (Claude, Devin,
Copilot Workspace, etc.). The same review standards apply:

- The PR description must show that the author *understood* the
  problem, not just pattern-matched
- Tests must pass
- The change must be appropriately surgical for the bug class
- Architectural choices must be motivated, not magic

If an AI PR doesn't meet the bar, mark it `needs-rework` with
specific feedback rather than closing — the agent can iterate.

## Code review expectations

- Reviewers respond within 7 days. If we haven't, ping us in the
  PR thread
- Maintainers may rebase your PR if main has moved
- Bigger PRs may get split into smaller mergeable pieces
- "Approved" means a maintainer wants this merged; we may still
  hold for second review on architectural changes

## Reporting bugs

Open an issue with:

1. **Steps to reproduce** (input files: source image + target
   video specs, not the files themselves)
2. **Expected vs actual behavior**
3. **Full traceback** if there's an exception
4. **Environment**: Python version, OS, GPU, CUDA version, output
   of `pip list`

## Reporting abuse

If you encounter this software being used in violation of
[USAGE_POLICY.md](USAGE_POLICY.md), open an issue tagged
`abuse-report` or contact the maintainers directly.

## License

By submitting a contribution, you agree to license it under the
Apache License 2.0 (matching the project license) and you confirm
that the contribution is your original work or appropriately
attributed if not.
