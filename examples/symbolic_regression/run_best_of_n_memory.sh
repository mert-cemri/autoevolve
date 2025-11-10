#!/bin/bash

# Run Best-of-N + Memory experiments for all symbolic regression problems
# Uses config_best_of_n_memory.yaml with memory enabled

# Define splits and their configurations
# Format: "split_name:count:prefix"
splits=(
    "bio_pop_growth:24:BPG"
    "chem_react:36:CRK"
    "matsci:25:MatSci"
    "phys_osc:44:PO"
)

base_problems_dir="./problems"
strategy_config="./config_best_of_n_memory.yaml"

# Verify strategy config exists
if [[ ! -f "$strategy_config" ]]; then
    echo "Error: Strategy config not found at $strategy_config"
    exit 1
fi

echo "Starting all Best-of-N + Memory experiments..."
echo "Using config: $strategy_config"
echo "Strategy: Best-of-N + Memory (n_lineages=4, memory enabled)"
echo ""

for split_config in "${splits[@]}"; do
    # Parse the configuration
    IFS=':' read -r split_name count problem_dir_prefix <<< "$split_config"

    echo ""
    echo "----------------------------------------------------"
    echo "Processing Split: $split_name (Best-of-N + Memory)"
    echo "Number of problems: $count"
    echo "Problem directory prefix: '$problem_dir_prefix'"
    echo "Expected problem path structure: $base_problems_dir/$split_name/${problem_dir_prefix}[ID]/"
    echo "----------------------------------------------------"

    # Loop from problem_id 0 to count-1
    for (( i=0; i<count; i++ )); do
        # Construct the path to the specific problem's directory
        # e.g., ./problems/chem_react/CRK0
        problem_dir="$base_problems_dir/$split_name/$problem_dir_prefix$i"

        initial_program_path="$problem_dir/initial_program.py"
        evaluator_path="$problem_dir/evaluator.py"

        # --- Sanity checks for file existence ---
        if [[ ! -f "$initial_program_path" ]]; then
            echo "  [Problem $i] SKIPPING: Initial program not found at $initial_program_path"
            continue
        fi
        if [[ ! -f "$evaluator_path" ]]; then
            echo "  [Problem $i] SKIPPING: Evaluator not found at $evaluator_path"
            continue
        fi
        # --- End Sanity checks ---

        echo "  Launching $split_name - Problem $i ($initial_program_path) with Best-of-N + Memory"
        # Run the experiment in the background with best-of-n flag and memory config
        cmd="python ../../openevolve-run.py \"$initial_program_path\" \"$evaluator_path\" --best-of-n --config \"$strategy_config\" --iterations 200"
        eval $cmd &
    done
    wait    # let's do split by split
done

echo ""
echo "All Best-of-N + Memory experiment processes have been launched in the background."
echo "Waiting for all background processes to complete..."
wait
echo ""
echo "All Best-of-N + Memory experiments have completed."
