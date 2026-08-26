# Data model

RetainRec operates on grounded preference evidence rather than a dataset's raw
storage format. Dataset-specific processing produces two UTF-8 JSON Lines files:
an item catalog and a chronological interaction stream. The rest of the code is
shared across datasets.

```text
data/processed/<dataset>/
├── items.jsonl
└── events.jsonl
```

This boundary is deliberate. Raw releases differ in feedback signals,
candidate construction, metadata, and licensing, while the retention algorithm
needs only ordered evidence, confidence, context, and item attributes. The
normalization step must use the dataset's frozen evidence map; the core loader
does not infer missing choices from unlabeled interactions.

## Item catalog

Each line in `items.jsonl` contains a stable item identifier and the attributes
available to the relation grammar:

```json
{"item_id":"m104","attributes":{"genre":["drama"],"year":"2021","director":"director-17","content_type":"film"}}
```

Attributes may be scalars or lists. Continuous values such as duration or
popularity should be bucketed during dataset processing, before users are split.
The resulting bucket identifiers remain fixed throughout an experiment.

## Preference events

Each line in `events.jsonl` represents one grounded pairwise observation:

```json
{"event_id":"u8-e91","user_id":"u8","position":91,"preferred_item_id":"m104","dispreferred_item_id":"m377","confidence":0.83,"context":{"content_type":"film"}}
```

Fields have the following meanings:

- `event_id`: globally unique event identifier;
- `user_id`: stable, de-identified user identifier;
- `position`: integer order within the user's stream;
- `preferred_item_id`, `dispreferred_item_id`: two distinct catalog items;
- `confidence`: dataset-specific evidence weight in `[0, 1]`;
- `context`: metadata used by exact contextual predicates.

The loader checks unique identifiers, chronological order, item references,
distinct pair members, and confidence bounds. Relation grounding then emits
support for the observed direction and contradiction for its inverse.

## Dataset grounding

For KuaiRand-1K, pairwise records are derived from randomized exposure and
observed engagement. Exposure probabilities affect the confidence code, while
category, semantic tags, duration, popularity, and official metadata provide
predicate values.

For InstructRec-MovieTV, chronological reviews or instructions provide the
preference direction and released candidate metadata provides genre, year,
director, and content attributes. Only directions grounded in the source record
are admitted.

The repository does not redistribute either raw dataset. Source locations and
licenses are maintained by the official KuaiRand and iAgent/InstructRec
repositories.

## Rolling transitions

Users are assigned to disjoint validation and test partitions before any
transition is materialized. The constructor scans each user chronologically and
creates a transition whenever a previously absent relation reaches the configured
support count.

Each transition contains four non-overlapping views:

- `support_evidence`: history available to the validity selector;
- `guard_cases`: pre-transition pairs used by the bounded updater;
- `audit_cases`: later pairs used only after an update decision;
- `future_evidence`: later observations used to define transition validity.

Identifiers and item identities are disjoint across the four views. Candidate
and pair ordering is based on canonical content hashes, so construction does not
depend on a selector, ranker, or update method.

Window sizes and grammar attributes are defined in `configs/datasets/`. Run the
loader before constructing manifests:

```bash
retainrec validate-data --data-dir data/processed/kuairand
```

The small records under `data/examples/smoke/` follow the same schema and are
intended for local development.

