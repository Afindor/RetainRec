# RetainRec

RetainRec studies a recurring failure mode in continually personalized
recommenders: an update can learn a new preference while silently breaking an
older behavior that is still useful. The project treats retention as a decision
made at each state transition. It estimates which previously satisfied
preference relations remain valid, protects only those relations during the
update, and chooses the smallest feasible policy change.

The implementation follows the same data path used in our experiments:

```text
grounded interaction stream
        │
        ├── relation evidence ──> validity selector ──> retention obligations
        │
current policy + new requirement ──> bounded proposals ──> guard ──> commit/reject
                                                               │
                                                               └── sealed audit
```

The selector and updater can use only information available at the transition.
Audit pairs and future evidence are opened after the state has been committed or
rejected. This boundary is enforced in the transition manifests and model
adapters rather than left to experiment convention.

## Quick start

RetainRec has no required third-party Python dependencies. A local installation
is enough for the deterministic example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
bash scripts/run_smoke.sh
```

The example contains four short synthetic user streams. It exercises rolling
transition construction, validation-only selector fitting, three-step bounded
updating, cross-version auditing, ledger verification, and aggregation. The
deterministic backend is useful for development and debugging; benchmark runs
use a model server.

## Codebase

```text
configs/                 Dataset windows, method settings, and run definitions
data/examples/smoke/     Synthetic interaction streams used by tests and examples
docs/                    Architecture, data, experiments, and extension guides
scripts/                 Common development and experiment commands
src/retainrec/           Construction, selection, updating, serving, and metrics
tests/                   Unit tests and an end-to-end pipeline test
```

The main modules are:

- `construction`: builds chronological, item-disjoint transition manifests;
- `grammar` and `evidence`: ground preference relations and aggregate support;
- `selector`: fits the calibrated validity model and selects obligations;
- `policy` and `updater`: represent edits and run the bounded proposal loop;
- `ranking`: connects policy states to deterministic, command, or vLLM rankers;
- `metrics`: computes acquisition and cross-version regression outcomes;
- `ledger`, `comparison`, and `reporting`: store and analyze experiment runs.

See [Architecture](docs/ARCHITECTURE.md) for the control flow and module
boundaries.

## Data

The experiment code consumes a small normalized representation: an item catalog
and a chronological stream of grounded pairwise preference events. Keeping this
representation separate from dataset-specific parsing makes the relation
grammar and transition constructor independent of raw file layouts.

The full schema and window construction are described in [Data](docs/DATA.md).
The included example records are synthetic. Raw benchmark data and model
checkpoints are intentionally kept outside the repository.

We use the official [KuaiRand](https://github.com/chongminggao/KuaiRand) and
[iAgent/InstructRec](https://github.com/WujiangXu/iAgent) releases. Their source
repositories remain the reference for downloads, licenses, and upstream file
formats.

## Running experiments

Dataset and method choices live in JSON configuration files rather than shell
scripts. A typical run consists of four commands:

1. validate the normalized data;
2. construct transition manifests;
3. fit the validity selector on validation users;
4. process test transitions chronologically and aggregate the ledger.

The commands, model-server setup, and output layout are collected in
[Experiments](docs/EXPERIMENTS.md). Published methods are connected through a
small process adapter so they keep their native state and update rules; see
[Method integration](docs/BASELINES.md).

## Development

```bash
python -m pip install -e '.[dev]'
bash scripts/run_checks.sh
```

The tests cover relation grounding, evidence decay, selector fitting, edit
classification, proposal bounds, label-blind serving requests, ledger tamper
detection, external method isolation, and a complete reference run. Guidance
for adding datasets, rankers, and update methods is in
[Development](docs/DEVELOPMENT.md).

