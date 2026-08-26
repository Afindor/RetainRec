# Architecture

RetainRec is organized as a sequence of pure transformations around two stateful
boundaries: model serving and the experiment ledger. This keeps construction,
selection, updating, and evaluation independently testable.

## Transition construction

`retainrec.construction` loads normalized records, partitions users, detects
relation triggers, and materializes rolling transition manifests.
`retainrec.grammar` instantiates pairwise and contextual relations from the
configured metadata inventory. Pair cases are ordered by canonical hashes and
formed without replacement.

`TransitionManifest.validate_boundaries` checks the information boundary before
a manifest is written. Support and guard information is available during the
update; audit and future records remain sealed until evaluation.

## Validity and obligation selection

`retainrec.evidence` computes confidence-weighted, time-decayed support and
contradiction. `ValidityEstimator` in `retainrec.selector` fits nonnegative
coefficients and a Platt calibration map using validation transitions.

At test time, selection combines four signals:

1. the calibrated probability that a relation remains valid;
2. the relative costs of missing a valid relation and retaining a stale one;
3. exact applicability of the relation's context gate;
4. whether the current state passes the relation's guard cases.

Selected relations become explicit obligations for one state transition.

## Policy updating

`retainrec.policy` represents a user policy as typed clauses and recognizes add,
delete, replace, and scope edits. Candidate distance combines normalized
operation cost with token-level Levenshtein distance.

`retainrec.updater` runs a fixed proposal budget. The first call receives the
current state, new requirements, and selected obligations. Later calls revise
the previous candidate using only unmet requirements and failed predicate
scopes. Every candidate is parsed, checked against the frozen relation universe,
and evaluated on the guard. The feasible candidate with minimum distance is
committed; if none is feasible, the original state is retained.

`DeterministicProposer`, `CommandPolicyProposer`, and `VLLMPolicyProposer`
implement the same interface for tests, local processes, and model serving.

## Ranking and sealed evaluation

`retainrec.ranking` exposes a single `Ranker` interface. Serving payloads include
the policy, context, candidate identifiers, and candidate metadata. They do not
include the preferred side of an audit pair.

After a commit or rejection, `retainrec.metrics` evaluates the old and new states
on identical audit requests. It computes acquisition rate, update regression
rate, obligation generalization rate, policy edit ratio, ranking metrics, and
common-pass diagnostics. Undefined denominators remain undefined.

## Run state and analysis

`ExperimentLedger` appends one record per seed and transition. Each record is
linked to its predecessor and includes input hashes, state hashes and
serializations, selector decisions, proposal outcomes, relation outcomes,
latency, and token usage.

`retainrec.pipeline` connects the construction, fitting, update, and aggregation
stages. `retainrec.comparison` aligns two ledgers by seed, user, and transition
for paired user-clustered inference. `retainrec.reporting` exports compact CSV
summaries from aggregate JSON files.

## Module guide

| Area | Modules | Main tests |
|---|---|---|
| Relation grounding | `grammar`, `models` | `test_grammar.py` |
| Evidence and validity | `evidence`, `selector` | `test_evidence_selector.py` |
| Policy editing | `policy`, `updater` | `test_policy_operations.py`, `test_updater_metrics.py` |
| Serving boundary | `ranking`, `controls` | `test_ranking_boundary.py`, `test_external_adapter.py` |
| Construction and execution | `construction`, `pipeline` | `test_construction_pipeline.py` |
| Storage and analysis | `ledger`, `metrics`, `comparison`, `reporting` | ledger and pipeline tests |

