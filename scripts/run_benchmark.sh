#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: run_benchmark.sh <kuairand|movietv> <normalized-data-dir> <artifact-dir> <vllm-endpoint> <model>" >&2
  exit 2
fi

dataset_name="$1"
data_dir="$2"
artifact_dir="$3"
vllm_endpoint="$4"
model_name="$5"

case "${dataset_name}" in
  kuairand|movietv) ;;
  *) echo "unsupported dataset: ${dataset_name}" >&2; exit 2 ;;
esac

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/retainrec-pycache"
if [[ -d "${artifact_dir}" ]] && [[ -n "$(find "${artifact_dir}" -mindepth 1 -print -quit)" ]]; then
  echo "artifact directory must be absent or empty: ${artifact_dir}" >&2
  exit 2
fi
if [[ ! -f "${data_dir}/items.jsonl" || ! -f "${data_dir}/events.jsonl" ]]; then
  echo "normalized data requires items.jsonl and events.jsonl: ${data_dir}" >&2
  exit 2
fi
mkdir -p "${artifact_dir}"

dataset_config="configs/datasets/${dataset_name}.json"
method_config="configs/methods/retainrec.json"
transitions="${artifact_dir}/transitions.jsonl"
selector="${artifact_dir}/selector.json"
ledger="${artifact_dir}/retainrec-ledger.jsonl"
aggregate="${artifact_dir}/retainrec-aggregate.json"

PYTHONPATH=src python3 -m retainrec.cli validate-data --data-dir "${data_dir}"

PYTHONPATH=src python3 -m retainrec.cli build-manifest \
  --dataset-config "${dataset_config}" \
  --data-dir "${data_dir}" \
  --output-dir "${artifact_dir}"

PYTHONPATH=src python3 -m retainrec.cli fit-selector \
  --dataset-config "${dataset_config}" \
  --method-config "${method_config}" \
  --transitions "${transitions}" \
  --output "${selector}"

PYTHONPATH=src python3 -m retainrec.cli run-test \
  --dataset-config "${dataset_config}" \
  --method-config "${method_config}" \
  --selector "${selector}" \
  --transitions "${transitions}" \
  --data-dir "${data_dir}" \
  --ledger "${ledger}" \
  --seeds 2024001 2024002 2024003 2024004 2024005 \
  --backend vllm \
  --endpoint "${vllm_endpoint}" \
  --model "${model_name}"

PYTHONPATH=src python3 -m retainrec.cli verify-ledger --ledger "${ledger}"
PYTHONPATH=src python3 -m retainrec.cli aggregate --ledger "${ledger}" --output "${aggregate}"
