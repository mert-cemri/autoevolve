#!/bin/bash

# Run Adaptive Exploration/Exploitation experiments for all symbolic regression problems
# Uses config_map_elites_adaptive.yaml with adaptive search enabled

# Define splits and their configurations
# Format: "split_name:count:prefix"
splits=(
    "bio_pop_growth:24:BPG"
    "chem_react:36:CRK"
    "matsci:25:MatSci"
    "phys_osc:44:PO"
)

base_problems_dir="./problems"
strategy_config="./config_map_elites_adaptive.yaml"

# Verify strategy config exists
if [[ ! -f "$strategy_config" ]]; then
    echo "Error: Strategy config not found at $strategy_config"
    exit 1
fi

echo "Starting all Adaptive Search experiments..."
echo "Using config: $strategy_config"
echo "Strategy: MAP-Elites with Adaptive Exploration/Exploitation"
echo "Adaptive: 10-70% exploration based on recent improvements"
echo "Models: gpt-4o (80%) + o3 (20%)"
echo "Iterations: 200 per problem"
echo ""

for split_config in "${splits[@]}"; do
    # Parse the configuration
    IFS=':' read -r split_name count problem_dir_prefix <<< "$split_config"

    echo ""
    echo "----------------------------------------------------"
    echo "Processing Split: $split_name (Adaptive)"
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

        echo "  Launching $split_name - Problem $i ($initial_program_path) with Adaptive Search"
        # Run the experiment in the background with adaptive config
        # Save to openevolve_output_adaptive to avoid overwriting base runs
        cmd="python ../../openevolve-run.py \"$initial_program_path\" \"$evaluator_path\" --config \"$strategy_config\" --output \"$problem_dir/openevolve_output_adaptive\" --iterations 200"
        eval $cmd &
    done
    wait    # let's do split by split
done

echo ""
echo "All Adaptive Search experiment processes have been launched in the background."
echo "Waiting for all background processes to complete..."
wait
echo ""
echo "All Adaptive Search experiments have completed."
