#!/usr/bin/env bash

# sweeps
yamls=('sweeps/general_sweeps/sweep_aggressive.yaml' 'sweeps/general_sweeps/sweep_arrogant.yaml' 'sweeps/general_sweeps/sweep_assertiv.yaml' 'sweeps/general_sweeps/sweep_collaborative.yaml' 'sweeps/general_sweeps/sweep_confident.yaml' 'sweeps/general_sweeps/sweep_dominant.yaml' 'sweeps/general_sweeps/sweep_enthusiastic.yaml' 'sweeps/general_sweeps/sweep_friendly.yaml' 'sweeps/general_sweeps/sweep_good_natured.yaml' 'sweeps/general_sweeps/sweep_independent.yaml' 'sweeps/general_sweeps/sweep_kind.yaml' 'sweeps/general_sweeps/sweep_leader_like.yaml' 'sweeps/general_sweeps/sweep_likeable.yaml' 'sweeps/general_sweeps/sweep_risk.yaml' 'sweeps/general_sweeps/sweep_sincere.yaml' 'sweeps/general_sweeps/sweep_warm.yaml')

# sweep_agents.txt file
agents_file="sweep_agents.txt"

# Create sweep agents
for yaml in "${yamls[@]}"; do
    wandb sweep $yaml
    done

# Fetch sweep agents
python fetch_sweep_agents.py 

# Read the file line by line
while IFS= read -r command
do
  # Extract the project name (SSU_MuSe2024_AGGRESSIVE) from the command
  project_name=$(echo "$command" | grep -oP '(?<=feelsgood_muse/)[^/]+')
  
  # Generate a unique session name using the project name and a hash of the full command
  session_name="${project_name}_$(echo "$command" | awk -F '/' '{print $NF}')"

  # Create a new tmux session and run the command
  tmux new-session -d -s "$session_name" "$command"
  
  echo "Started tmux session '$session_name' with command: $command"
done < "$agents_file"

rm -f "$agents_file"