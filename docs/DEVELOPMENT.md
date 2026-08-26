# Development

## Local environment

The core package uses the Python standard library. Tests and linting are kept in
the optional development dependency group:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
bash scripts/run_checks.sh
```

Generated run directories, local datasets, checkpoints, caches, and virtual
environments are ignored by Git.

## Adding a dataset

1. Produce `items.jsonl` and `events.jsonl` using the schema in `DATA.md`.
2. Add a dataset configuration under `configs/datasets/`.
3. Validate the normalized directory with `retainrec validate-data`.
4. Build manifests and inspect the validation/test transition counts.
5. Add a compact synthetic case when the dataset introduces a new attribute or
   contextual pattern.

Dataset-specific parsing should end at the normalized record boundary. Relation
grounding, user splitting, rolling windows, and canonical pairing belong in the
shared constructor.

## Adding a ranker or proposer

A ranker implements `Ranker.rank(state, case)` and returns every candidate
identifier exactly once. Target labels must not be sent to the serving process.
Invalid, duplicate, or incomplete rankings are treated as failures.

A proposer implements `PolicyProposer.propose`. It may use the current state,
the previous candidate, active requirements, selected relations, and failed
predicate scopes. It must return a state for the same user, increment the state
version exactly once, stay within the clause limit, and use only relations in
the supplied universe.

New serving adapters should record latency and token usage when the backend
reports them.

## Adding an update method

Small text-state variants can be implemented in `retainrec.controls`. Methods
with their own dependencies or state representation should run through
`ExternalMethodAdapter`. This keeps upstream optimizers, prompts, and training
loops intact while sharing transition manifests and behavioral evaluation.

Use validation users for operating-configuration search. Test ledgers should be
written once to a new directory; the runner refuses to append to an existing
ledger.

## Code conventions

- Keep source comments and public documentation in English.
- Prefer typed immutable records at module boundaries.
- Make randomness explicit through configuration or function arguments.
- Preserve undefined metric denominators instead of coercing them to zero.
- Do not expose audit or future records to an update method.
- Keep generated outputs outside source and configuration directories.
- Add a focused test for every new information boundary or state transition.

