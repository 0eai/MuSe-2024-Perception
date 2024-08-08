import wandb

# Authenticate with wandb
wandb.login()

# List of project names
project_names = ["MuSe2024_WARM","MuSe2024_SINCERE","MuSe2024_RISK","MuSe2024_LIKEABLE","MuSe2024_LEADER-LIKE","MuSe2024_KIND","MuSe2024_INDEPENDENT","MuSe2024_GOOD-NATURED","MuSe2024_FRIENDLY","MuSe2024_ENTHUSIASTIC","MuSe2024_DOMINANT","MuSe2024_CONFIDENT","MuSe2024_COLLABORATIVE","MuSe2024_ASSERTIV","MuSe2024_ARROGANT","MuSe2024_AGGRESSIVE"]
entity_name = 'feelsgood_muse'

# Define an empty string to store sweep information
sweep_data = ""
for idx, project_name in enumerate(project_names):
    print(project_name, idx, len(project_names))
    
    api = wandb.Api()
    # Get all sweeps
    sweeps = api.project(project_name, entity=entity_name).sweeps()
    print(f'sweeps: {sweeps}')
    sweep_info = f"wandb agent {entity_name}/{project_name}/{sweeps[0].id}\n"
    sweep_data += sweep_info
    
# Save sweep data to a text file (assuming 'outs' directory exists)
if sweep_data:
  with open("sweep_agents.txt", "w") as f:
    f.write(sweep_data)
  print("Sweep information saved to sweep_agents.txt")
else:
  print("No sweeps found for any projects.")