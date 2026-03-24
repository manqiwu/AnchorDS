#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_anchor_quickstart.sh
#   bash scripts/run_anchor_quickstart.sh "a bald eagle carved out of wood"
#   bash scripts/run_anchor_quickstart.sh "a bald eagle carved out of wood" ip-gs

PROMPT="${1:-a bald eagle carved out of wood}"
TARGET="${2:-all}"
GPU="${GPU_ID:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/outputs/quickstart_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# Keep command list aligned with the AnchorDS README quickstart matrix.
declare -a RUNS=(
  "ip-gs|configs/gaussiandreamer-sd1.5-anchorDS-ipadapter-finetune.yaml"
  "ip-nerf|configs/nerf-sd1.5-anchorDS-ipadapter-finetune.yaml"
  "cn-gs|configs/gaussiandreamer-sd2.1-anchorDS-controlnet-finetune.yaml"
  "cn-nerf|configs/nerf-sd2.1-anchorDS-controlnet-finetune.yaml"
  "sds-gs|configs/gaussiandreamer-sd1.5-sds.yaml"
  "sdsb-nerf|configs/nerf-sd1.5-sds_bridge.yaml"
)

run_one() {
  local name="$1"
  local cfg="$2"
  local log_file="${LOG_DIR}/${name}.log"

  echo "========== [${name}] START $(date '+%F %T') ==========" | tee -a "${log_file}"
  echo "config=${cfg}" | tee -a "${log_file}"
  echo "gpu=${GPU}" | tee -a "${log_file}"
  echo "prompt=${PROMPT}" | tee -a "${log_file}"

  (
    cd "${ROOT_DIR}"
    python launch.py \
      --config "${cfg}" \
      --train \
      --gpu "${GPU}" \
      system.prompt_processor.prompt="${PROMPT}"
  ) 2>&1 | tee -a "${log_file}"

  echo "========== [${name}] END   $(date '+%F %T') ==========" | tee -a "${log_file}"
}

print_targets() {
  printf "Available targets:\n"
  for item in "${RUNS[@]}"; do
    printf "  - %s\n" "${item%%|*}"
  done
  printf "  - all\n"
}

if [[ "${TARGET}" == "list" ]]; then
  print_targets
  exit 0
fi

matched=0
for item in "${RUNS[@]}"; do
  name="${item%%|*}"
  cfg="${item#*|}"

  if [[ "${TARGET}" == "all" || "${TARGET}" == "${name}" ]]; then
    matched=1
    run_one "${name}" "${cfg}"
  fi
done

if [[ "${matched}" -eq 0 ]]; then
  echo "Unknown target: ${TARGET}"
  print_targets
  exit 1
fi

echo "All done. Logs are in: ${LOG_DIR}"
