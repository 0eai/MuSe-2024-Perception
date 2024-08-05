import os
from pathlib import Path
import torch
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# adjust your paths here.
BASE_PATH = "/mnt/sda/datasets/MuSe/2024/"
PERCEPTION_PATH = os.path.join(BASE_PATH, 'c1_muse_perception')
HUMOR_PATH = os.path.join(BASE_PATH, 'c2_muse_humor')

PERCEPTION = 'perception'
HUMOR = 'humor'

RNN = 'RNN'
TRANSFORMER = 'TF'

ENCODERS = [RNN, TRANSFORMER]

TASKS = [PERCEPTION, HUMOR]

PATH_TO_FEATURES = {
    PERCEPTION: os.path.join(PERCEPTION_PATH, 'feature_segments'),
    HUMOR: os.path.join(HUMOR_PATH, 'feature_segments')
}

# humor is labelled every 2s, but features are extracted every 500ms
N_TO_1_TASKS = {HUMOR, PERCEPTION}

ACTIVATION_FUNCTIONS = {
    PERCEPTION: torch.nn.Sigmoid,
    HUMOR: torch.nn.Sigmoid
}

NUM_TARGETS = {
    HUMOR: 1,
    PERCEPTION: 1
}


PATH_TO_LABELS = {
    PERCEPTION: os.path.join(PERCEPTION_PATH, 'labels.csv'),
    HUMOR: os.path.join(HUMOR_PATH, 'label_segments')
}

PATH_TO_METADATA = {
    PERCEPTION: os.path.join(PERCEPTION_PATH, 'metadata'),
    HUMOR: os.path.join(HUMOR_PATH, 'metadata')
}

PARTITION_FILES = {task: os.path.join(path_to_meta, 'partition.csv') for task,path_to_meta in PATH_TO_METADATA.items()}

PERCEPTION_LABELS = ['assertiv','competent','dominant','confident','independent','enthusiastic','good_natured','sincere','collaborative','friendly','forceful','aggressive','expressive','likeable','trustworthy','intelligent','arrogant','emotional','yielding','naive','competitive','leader_like','productive','sympathetic','kind','charismatic','compassionate','warm','understanding','risk','attractive','envious','pity','angry','admiring']

MULTI_ATT_AMIABLE = 'collaborative-friendly-good_natured-kind-likeable-sincere-warm'
MULTI_ATT_FORCEFUL = 'aggressive-assertiv-dominant'
MULTI_ATT_SELFASSURED = 'confident-independent-leader_like'

current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:23]

EXP_PATH = '~/MuSe-2024/outs'

OUTPUT_PATH = os.path.join(EXP_PATH, 'results')
LOG_FOLDER = os.path.join(OUTPUT_PATH, 'log_muse')
DATA_FOLDER = os.path.join(OUTPUT_PATH, 'data_muse')
MODEL_FOLDER = os.path.join(OUTPUT_PATH, 'model_muse')
PREDICTION_FOLDER = os.path.join(OUTPUT_PATH, 'prediction_muse')

BEST_MODEL_FOLDER = os.path.join(EXP_PATH, 'best_models')
