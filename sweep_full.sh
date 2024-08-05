#!/usr/bin/env bash

yamls=('sweeps/sweep_aggressive.yaml' 'sweeps/sweep_arrogant.yaml' 'sweeps/sweep_assertiv.yaml' 'sweeps/sweep_collaborative.yaml' 'sweeps/sweep_confident.yaml' 'sweeps/sweep_dominant.yaml' 'sweeps/sweep_enthusiastic.yaml' 'sweeps/sweep_friendly.yaml' 'sweeps/sweep_good_natured.yaml' 'sweeps/sweep_independent.yaml' 'sweeps/sweep_kind.yaml' 'sweeps/sweep_leader_like.yaml' 'sweeps/sweep_likeable.yaml' 'sweeps/sweep_risk.yaml' 'sweeps/sweep_sincere.yaml' 'sweeps/sweep_warm.yaml')

# # Read the sweep_log.txt file
agents_file="outs/sweep_agents.txt"

rm -f "$agents_file"

for yaml in "${yamls[@]}"; do
    wandb sweep $yaml
    done

python get_sweep.py 

# Read the file line by line
while IFS= read -r command
do
  # Extract the project name (SSU_MuSe2024_AGGRESSIVE) from the command
  project_name=$(echo "$command" | grep -oP '(?<=ssu/)[^/]+')
  
  # Generate a unique session name using the project name and a hash of the full command
  session_name="${project_name}_$(echo "$command" | awk -F '/' '{print $NF}')"

  # Create a new tmux session and run the command
  tmux new-session -d -s "$session_name" "$command"
  
  echo "Started tmux session '$session_name' with command: $command"
done < "$agents_file"