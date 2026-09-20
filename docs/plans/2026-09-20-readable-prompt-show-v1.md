# PromptManager — Readable `prompt-show` Text View v1

Status: completed
Owner: PromptManager Team

## Goal

Make the default text form of `prompt-show` a readable, copy-friendly operator view while retaining JSON contracts unchanged.

## Confirmed baseline

- Current text output is a flat `key: value` list followed by unbounded context; an actual video prompt emitted about 11 KB with a one-line description and little visual hierarchy.
- The existing default already includes the full `context`; this slice improves presentation rather than changing that disclosure choice.
- `--json` remains the compact structured read model and `--json --full` remains the complete persisted-record opt-in. Neither JSON mode changes here.

## Delivered text contract

- Default text output has a title/header, ordered metadata rows, a wrapped Description section, and a Prompt body section.
- When a non-empty context exists, it is enclosed exactly as:

  ```text
  <prompt_body>

  ...raw prompt context...

  </prompt_body>
  ```

  Blank separator lines make selection and copy/paste unambiguous.
- No `--summary` or `--body` flags are introduced in this slice; those are distinct behavior choices needing separate operator evidence.
- Prompt resolution by UUID or exact unique name remains unchanged.

## Execution order

1. Add RED text-output regressions for heading, metadata, wrapping, and exact body delimiters.
2. Add a private text renderer in the CLI command module and keep JSON logic intact.
3. Update runtime help and the developer CLI index.
4. Run focused tests, an isolated real CLI smoke with a long context, Ruff, format, CI-scope Pyright, and diff hygiene.

## Out of scope

- JSON shape/size changes.
- `--summary`, `--body`, pagination, terminal colour, or file output.
- Prompt content migration or console-encoding changes.
- Commit or push.

## Completion update

Completed 2026-09-20:

- Replaced flat default text output with a readable title/header, aligned metadata rows, and a wrapped Description section.
- Enclosed every non-empty text context in an exact blank-line-separated copy block:

  ```text
  <prompt_body>

  ...raw prompt context...

  </prompt_body>
  ```

- Preserved JSON and `--json --full` contracts unchanged.
- Updated runtime help, the developer CLI index, and changelog.

Verification completed:

- selected docs/CLI/domain regression suite: `144 passed`;
- isolated real CLI smoke with a long description and multiline context: passed; output contained the exact body delimiters and preserved `{{input_text}}` unchanged;
- Ruff lint and format: passed;
- CI-scope Pyright (`main.py config models`): `0 errors`;
- `git diff --check`: passed;
- `prompt-show --help` contract: passed.

Deferred unchanged: `--summary`, `--body`, terminal colour/paging, JSON-body volume, output files, and console-encoding investigation.
