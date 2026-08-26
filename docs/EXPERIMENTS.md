# Experiments

RetainRec experiments run as a sequence of explicit stages. Intermediate files
make it possible to inspect a transition stream, reuse it across methods, and
separate validation decisions from chronological test updates.

## Configuration

Configuration is split by responsibility:

- `configs/datasets/` defines relation attributes, user splits, trigger counts,
  window lengths, pair limits, and candidate counts;
- `configs/methods/retainrec.json` defines proposal, edit, guard, policy, token,
  and cost settings;
- `configs/experiments/` names the data directory, output directory, backend,
  and random seeds;
- `configs/baselines/registry.json` records method families used by comparison
  runs.

The KuaiRand and MovieTV experiment configurations use seeds `2024001` through
`2024005`. Dataset-specific windows stay in their dataset files rather than in
the runner.

## Building a transition stream

First validate the normalized records:

```bash
retainrec validate-data --data-dir data/processed/kuairand
```

Then construct rolling transitions:

```bash
retainrec build-manifest \
  --dataset-config configs/datasets/kuairand.json \
  --data-dir data/processed/kuairand \
  --output-dir artifacts/kuairand/retainrec
```

This writes `transitions.jsonl` and a compact manifest summary. Construction
includes the user partition, relation triggers, canonical candidate order, and
support/guard/audit/future boundary checks.

## Fitting the selector

```bash
retainrec fit-selector \
  --dataset-config configs/datasets/kuairand.json \
  --method-config configs/methods/retainrec.json \
  --transitions artifacts/kuairand/retainrec/transitions.jsonl \
  --output artifacts/kuairand/retainrec/selector.json
```

The selector stage reads validation transitions only. It chooses the evidence
decay, fits validity coefficients, calibrates probabilities, and selects the
stale-to-miss cost ratio. The output stores hashes of the dataset configuration,
method configuration, and transition file. The test runner stops if any of
those inputs changes.

## Running chronological updates

Start an OpenAI-compatible vLLM server with the desired model revision, then
run the convenience script:

```bash
bash scripts/run_benchmark.sh \
  kuairand \
  data/processed/kuairand \
  artifacts/kuairand/retainrec \
  http://127.0.0.1:8000 \
  meta-llama/Llama-3.2-3B-Instruct
```

The runner processes each test user's transitions in order for all configured
seeds. With `proposal_revisions=2`, every transition makes three proposal calls,
including failed parses. Later calls receive the previous candidate, unmet
requirements, and failed predicate scopes, but not item-level guard failures,
audit cases, or future evidence.

The output directory must be absent or empty. A run writes:

```text
transitions.jsonl
manifest.json
selector.json
retainrec-ledger.jsonl
retainrec-aggregate.json
```

## Inspecting runs

Verify a ledger before analysis:

```bash
retainrec verify-ledger \
  --ledger artifacts/kuairand/retainrec/retainrec-ledger.jsonl
```

Regenerate an aggregate JSON file:

```bash
retainrec aggregate \
  --ledger artifacts/kuairand/retainrec/retainrec-ledger.jsonl \
  --output artifacts/kuairand/retainrec/retainrec-aggregate.json
```

Export several aggregate files to one CSV:

```bash
retainrec export-summary \
  --inputs artifacts/kuairand/retainrec/retainrec-aggregate.json \
  --output artifacts/tables/continual-summary.csv
```

Compare two methods on aligned seed-user-transition coordinates:

```bash
retainrec compare-ledgers \
  --left artifacts/kuairand/retainrec/retainrec-ledger.jsonl \
  --right artifacts/kuairand/uniform-guard/ledger.jsonl \
  --output artifacts/comparisons/retainrec-vs-uniform-guard.json
```

The comparison reports the primary paired difference, a user-clustered
bootstrap interval, common-pass URR, and common-pass coverage. Aggregation first
combines relation outcomes within a transition, transitions within a user, and
seed means within a user before averaging users.

## Local reference run

```bash
bash scripts/run_smoke.sh
```

The reference run uses the same construction, fitting, update, ledger, and
analysis code as a model-backed run. Its deterministic ranking behavior is
intended for fast development checks rather than empirical comparison.

