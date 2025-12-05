#!/bin/bash
# Master script to run all adaptive exploration experiments
# This will run each example sequentially or you can run them in parallel

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
RUN_MODE="sequential"  # sequential or parallel
EXAMPLES="all"  # all, circle_packing, prism, llm_sql, signal_processing

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--parallel)
            RUN_MODE="parallel"
            shift
            ;;
        -e|--example)
            EXAMPLES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -p, --parallel        Run all examples in parallel"
            echo "  -e, --example NAME    Run specific example (circle_packing, prism, llm_sql, signal_processing)"
            echo "                        Use 'all' to run all examples (default)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run all examples sequentially"
            echo "  $0 -p                 # Run all examples in parallel"
            echo "  $0 -e circle_packing  # Run only circle_packing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "OpenEvolve - Adaptive Exploration Experiments"
echo "=============================================="
echo "Mode: ${RUN_MODE}"
echo "Examples: ${EXAMPLES}"
echo "=============================================="

# Function to run an example
run_example() {
    local example_name=$1
    local script_path="${SCRIPT_DIR}/${example_name}/run_adaptive_exploration.sh"

    if [ -f "${script_path}" ]; then
        echo ""
        echo ">>> Starting ${example_name}..."
        bash "${script_path}"
        echo ">>> Finished ${example_name}"
    else
        echo "Warning: Script not found for ${example_name}: ${script_path}"
    fi
}

# Build list of examples to run
EXAMPLE_LIST=()
if [ "${EXAMPLES}" = "all" ]; then
    EXAMPLE_LIST=("circle_packing" "prism" "llm_sql" "signal_processing")
else
    EXAMPLE_LIST=("${EXAMPLES}")
fi

# Run examples
if [ "${RUN_MODE}" = "parallel" ]; then
    echo "Running examples in parallel..."
    pids=()
    for example in "${EXAMPLE_LIST[@]}"; do
        run_example "${example}" &
        pids+=($!)
    done

    # Wait for all processes
    echo "Waiting for all experiments to complete..."
    for pid in "${pids[@]}"; do
        wait $pid
    done
else
    echo "Running examples sequentially..."
    for example in "${EXAMPLE_LIST[@]}"; do
        run_example "${example}"
    done
fi

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "=============================================="
