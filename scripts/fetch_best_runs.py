import wandb
import itertools
from datetime import datetime
from dateutil import tz
import pandas as pd

def run_meets_criteria(run, filters):
    config_match = True
    for key, value in filters.items():
        config_match = run.config.get(key) == value

        if not config_match:
            break

    return  config_match

# wandb.login(key='')


api = wandb.Api()

RESULT_DIR = './results'

attributes = ['AGGRESSIVE', 'ARROGANT', 'ASSERTIV', 'CONFIDENT', 'DOMINANT', 'INDEPENDENT', 'RISK', 'LEADER-LIKE', 
              'COLLABORATIVE', 'ENTHUSIASTIC', 'FRIENDLY', 'GOOD-NATURED', 'KIND', 'LIKEABLE', 'SINCERE', 'WARM', 
              ]

entity_name = 'ssu'
project_prefix = "MuSe2024_"

metric_name = 'best_val_score'
metric_goal = 'maximize'

filters = {
        # 'encoder': 'RNN',
        # 'feature': 'vit-fer'
        # 'sweep': ''
          }


df = pd.DataFrame(columns=['encoder', 'feature'] + attributes)

encoders = ['RNN']
features = [ 'ds', 'egemaps', 'w2v-msp', 'faus', 'facenet512', 'vit-fer', 'bert-b', 'bert-l', 'roberta-sent', 'roberta-twt-sent', 'canine-c', 'distilroberta-emo', 'roberta-go-emo' ]

filter_combinations = itertools.product(encoders, features)
filters = {}

df = pd.DataFrame(columns=['encoder', 'feature'] + attributes)
df_run = pd.DataFrame(columns=['encoder', 'feature'] + attributes)

for comb in filter_combinations:
    print(f'Combination: {comb}')
    filters['encoder'] = comb[0]
    filters['feature'] = comb[1]

    row = {key: 0 for key in df.columns}
    row_run = {key: '' for key in df.columns}
    for key, value in filters.items():
        row[key] = value
        row_run[key] = value
        

    for att in attributes:
        print(f'\tAttribute: {att}')
        project_name = project_prefix + att
        runs = api.runs(f'{entity_name}/{project_name}')

        metric_name = 'best_val_score'
        metric = None
        best_run = None
        best_metric = 0


        for run in runs:
            print(f'\t\t\tRun[ id: {run.id} | name: {run.name} | status: {run.state}]')
            if run_meets_criteria(run, filters) and metric_name in run.summary.keys():
                metric = run.summary[metric_name]

            if metric is not None and ((metric_goal == 'maximize' and metric > best_metric)  or (metric_goal == 'minimize' and metric < best_metric)):
                    best_metric = metric
                    best_run = run

        if best_run:
            print(f"Project Name: {project_name}")
            print(f"Best run Name: {best_run.name}")
            print(f"Best metric value: {best_metric}")
            print(f"Run details: {best_run.url}")

            row[att.upper()] = best_metric
            row_run[att.upper()] = best_run.name
            print(row)
        else:
            print("No runs found with the specified metric.")
        print('='*50)
    df = df.append(row, ignore_index=True)
    df_run = df_run.append(row_run, ignore_index=True)

# Calculate max values for each attribute and append to DataFrame
max_row = {key: 'MAX' for key in df.columns[:2]}
for att in attributes:
    max_row[att] = df[att].max()

df = df.append(max_row, ignore_index=True)

time = datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M")
df.to_csv(f'{RESULT_DIR}/results_{time}_local-best.csv', index=False)
df_run.to_csv(f'{RESULT_DIR}/results_{time}_local-best-run.csv', index=False)