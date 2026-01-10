#!/bin/bash
#
# Run all VLLM examples with timing and status reporting
#
# Usage:
#   ./examples/run_all.sh
#   VLLM_RUN_TIMEOUT_SECONDS=300 ./examples/run_all.sh
#   VLLM_RUN_TIMEOUT_SECONDS=0 ./examples/run_all.sh  # No timeout
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Cleanup function to kill all child processes
cleanup() {
    # Kill all child processes of this script
    pkill -P $$ 2>/dev/null || true
    sleep 0.1
    pkill -9 -P $$ 2>/dev/null || true
    echo -e "\n${RED}Interrupted by user${NC}"
    exit 130
}

# Trap Ctrl+C and exit cleanly
trap cleanup INT TERM

# Configuration
TIMEOUT_SECONDS="${VLLM_RUN_TIMEOUT_SECONDS:-120}"
MIX_ENV_NAME="${VLLM_MIX_ENV:-prod}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Examples to run (in order)
EXAMPLES=(
    "basic.exs"
    "sampling_params.exs"
    "chat.exs"
    "batch_inference.exs"
    "structured_output.exs"
    "quantization.exs"
    "multi_gpu.exs"
    "embeddings.exs"
    "lora.exs"
    "timeout_config.exs"
    "direct_api.exs"
)

# Statistics
PASSED=0
FAILED=0
SKIPPED=0
declare -a FAILED_EXAMPLES

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}       VLLM Examples Test Runner       ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Project directory: ${PROJECT_DIR}"
echo -e "Timeout per example: ${TIMEOUT_SECONDS}s (0 = disabled)"
echo -e "Mix environment: ${MIX_ENV_NAME}"
echo -e "Examples to run: ${#EXAMPLES[@]}"
echo ""

cd "$PROJECT_DIR"

# Function to run a single example
run_example() {
    local example="$1"
    local example_path="examples/${example}"

    if [[ ! -f "$example_path" ]]; then
        echo -e "${YELLOW}SKIP${NC} ${example} (file not found)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    echo -e "${BLUE}----${NC} Running ${example} ${BLUE}----${NC}"

    local start_time=$(date +%s.%N)
    local exit_code=0

    if [[ "$TIMEOUT_SECONDS" -gt 0 ]]; then
        MIX_ENV="${MIX_ENV_NAME}" timeout --foreground "${TIMEOUT_SECONDS}s" mix run "$example_path" || exit_code=$?
    else
        MIX_ENV="${MIX_ENV_NAME}" mix run "$example_path" || exit_code=$?
    fi

    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "?")

    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}PASS${NC} ${example} (${duration}s)"
        PASSED=$((PASSED + 1))
    elif [[ $exit_code -eq 124 ]]; then
        echo -e "${YELLOW}TIMEOUT${NC} ${example} (exceeded ${TIMEOUT_SECONDS}s)"
        FAILED_EXAMPLES+=("$example (timeout)")
        FAILED=$((FAILED + 1))
    else
        echo -e "${RED}FAIL${NC} ${example} (exit code: ${exit_code})"
        FAILED_EXAMPLES+=("$example")
        FAILED=$((FAILED + 1))
    fi

    echo ""
}

# Run all examples
for example in "${EXAMPLES[@]}"; do
    run_example "$example"
done

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}              Summary                  ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Total:   ${#EXAMPLES[@]}"
echo -e "${GREEN}Passed:  ${PASSED}${NC}"
echo -e "${RED}Failed:  ${FAILED}${NC}"
echo -e "${YELLOW}Skipped: ${SKIPPED}${NC}"

if [[ ${#FAILED_EXAMPLES[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}Failed examples:${NC}"
    for ex in "${FAILED_EXAMPLES[@]}"; do
        echo -e "  - ${ex}"
    done
fi

echo ""

# Exit with appropriate code
if [[ $FAILED -gt 0 ]]; then
    exit 1
else
    echo -e "${GREEN}All examples passed!${NC}"
    exit 0
fi
