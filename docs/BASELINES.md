# Method integration

RetainRec evaluates different update mechanisms through a shared behavioral
interface. A method receives the state and information available at the current
transition, returns a new state, and is then evaluated on the same sealed
requests as every other method. Its internal representation can remain a model,
adapter, memory, or natural-language policy.

## External update methods

`ExternalMethodAdapter` runs an upstream implementation as a local process. The
process reads one JSON object from standard input and writes one JSON object to
standard output. This keeps environment-specific dependencies out of the core
package and avoids translating every method into RetainRec's policy format.

Input contains:

- the method name and current state;
- the user, transition, and chronological position;
- support evidence available by that position;
- newly triggered requirements and the transition context.

It does not contain audit cases, future evidence, selector decisions, guard
outcomes, or aggregate measurements.

A text-state method can return:

```json
{
  "state": {"user_id":"u8","version":4,"clauses":[]},
  "usage": {"latency_ms":1842.5,"input_tokens":3810,"output_tokens":281},
  "native_metadata": {"update_steps":8}
}
```

Model and adapter methods may use a family-specific state serializer while
keeping the same process boundary. Behavioral comparison happens through the
ranker and sealed requests, not through internal RetainRec predicates.

The method inventory in `configs/baselines/registry.json` groups the experiment
methods by state family. Upstream launch commands and native hyperparameter
grids belong to their own environments and are supplied when a run is started.

## Controlled update rules

`retainrec.controls` contains small update rules used to study individual design
choices:

- `no_update` leaves the state unchanged;
- `direct_policy_rewrite` applies the new requirement without retention;
- `uniform_guard_update` protects a fixed number of canonically ordered
  pre-passing relations;
- `regression_replay_update` adds relations observed in recent failures without
  predicting transition validity.

These controls share RetainRec's text-state representation because their role is
to change one part of the update logic. The deterministic ranker and proposer
serve a different purpose: they make pipeline behavior testable without a model
server.

## Operating configurations

Each method searches only over its native adaptation coordinates. Examples
include update learning rate and epochs for continual models, update steps and
regularization for adapters, or proposal budgets and acceptance thresholds for
text-state methods. Dataset windows, candidate manifests, model architecture,
and evaluation thresholds stay fixed.

Configuration selection is performed on validation users. The selected setting
is the one nearest the configured acquisition target while satisfying the
quality constraint; ties prefer higher validation quality and then lower update
cost. Once chosen, the setting is reused unchanged for the chronological test
stream.

