# PromptManager AGENTS.md

Operational instructions for AI agents working in this repository.

## Scope
- Work only within the requested task and this repository.
- Prefer small, bounded changes over broad cleanup.
- Before changing multiple files or behavior outside the requested slice, verify scope from code and current repo state.

## Working style
- Optimize for correctness, clarity, reliability, and maintainability over speed.
- Address root causes, not only symptoms.
- Prefer simple, explicit solutions over clever ones.
- Do not guess repository state, runtime behavior, or configuration; verify with tools.
- Report uncertainty explicitly as `do weryfikacji`.

## Repo facts
- Python: `3.13+`
- Environment: use the project virtualenv at `.venv/`
- Formatting and linting: `ruff`
- Type checking: `pyright` in strict mode
- Tests: `pytest`
- Coverage target: `>= 80%` for changed/new code; maintain overall gate parity when relevant

## Canonical commands
Run commands from the repo root.

```bash
.venv/bin/ruff check --fix .
.venv/bin/ruff check .
.venv/bin/ruff format .
.venv/bin/pyright <modified_files>
.venv/bin/pytest
```

For CI parity when needed:

```bash
.venv/bin/pytest -n auto --cov=core --cov-report=term-missing --cov-fail-under=80
```

## Code rules
- Keep functions and modules readable; refactor when complexity grows.
- Prefer typed structures over loose dict plumbing where that improves clarity.
- Use project configuration patterns already present in the repo instead of introducing parallel mechanisms.
- Never commit secrets or hardcode credentials.
- Validate and sanitize external/user input before templating, execution, or persistence.
- Never use bare `except:`.

## Documentation rules
- Keep documentation changes minimal and specific to the delivered slice.
- If shipped behavior changes, update the relevant docs and `docs/CHANGELOG.md`.
- Keep `README.md` user-focused and concise.
- Keep deeper developer detail in `docs/README-DEV.md` or focused docs, not by bloating this file.
- Existing docstring/history conventions in the codebase must be preserved when editing affected files.

## Testing and done criteria
A change is not done until the relevant verification passes.

Minimum expectations:
1. Run targeted validation for the changed area.
2. Run `ruff check .` after fixes.
3. Run `pyright` on modified files when type impact exists.
4. Run `pytest` for affected tests; run broader suite when risk or coupling is higher.
5. If behavior changed, verify docs are aligned.

## Boundaries
- **Always do**
  - verify current repo state before risky edits
  - keep changes bounded to the requested slice
  - prefer evidence from code/tests over assumptions
- **Ask first**
  - broad refactors
  - dependency changes
  - destructive cleanup
  - changes to repo-wide process/docs policy
- **Never do**
  - invent results without verification
  - overwrite unrelated local changes
  - commit secrets or credentials
  - treat this file as a project handbook; put long-form policy in docs

## Repo state hygiene
- Check working tree status before edits that may overlap with existing local changes.
- If the tree is dirty, avoid rewriting unrelated files and call out overlap risk.
- Prefer minimal diffs that preserve the user’s in-flight work.

## Note on long-form policy
This file is intentionally concise. Long-form engineering policy, product rules, or memory workflow should live in dedicated docs under `docs/` rather than here.
