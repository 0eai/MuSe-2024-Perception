## Setup Instructions

Follow these steps to set up the environment and run the scripts:

### 1. Clone the Repository

First, navigate to the parent directory where you want to clone the repository and run the following commands:

```bash
git clone https://github.com/0eai/MultimodalFeatureExtractor.git
cd MultimodalFeatureExtractor
```

### 2. Create and Activate a Virtual Environment

You can choose between using `venv` or `conda` to create a virtual environment.

#### Using `venv`

To avoid conflicts with other packages, it's recommended to create a virtual environment. Run the following commands to create and activate the virtual environment:

##### For Windows

```bash
python -m venv multimodal_env
.\multimodal_env\Scripts\activate
```

##### For macOS and Linux

```bash
python3 -m venv multimodal_env
source multimodal_env/bin/activate
```

#### Using `conda`

If you prefer using `conda`, run the following commands:

```bash
conda create --name multimodal_env python=3.8
conda activate multimodal_env
```

### 3. Install the Required Packages

Install the necessary Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Update the `BASE_PATH` in `config.py`

Before running the scripts, you need to update the `BASE_PATH` in the `config.py` file to point to your dataset location. Open the `config.py` file and modify the `BASE_PATH` variable:

```python
# config.py
BASE_PATH = "/mnt/sda/datasets/MuSe/2024/"
```

Replace `"/mnt/sda/datasets/MuSe/2024/"` with the path to your dataset.

### 5. Run the Scripts

Execute the main script for the perception task using different models and configurations. Below are some examples:

#### Using BERT Base Model

```bash
python scripts/main.py --task perception --modality transcriptions --hf_model google-bert/bert-base-cased --feature_segment bert-b --step_size 0.5 --device cuda
```

#### Using BERT Large Model

```bash
python scripts/main.py --task perception --modality transcriptions --hf_model google-bert/bert-large-cased --feature_segment bert-l --step_size 0.5 --device cuda
```

#### Using CANINE Model

```bash
python scripts/main.py --task perception --modality transcriptions --hf_model google/canine-c --feature_segment canine-c --step_size 0.5 --device cuda
```

#### Using RoBERTa Model for Sentiment Analysis

```bash
python scripts/main.py --task perception --modality transcriptions --hf_model siebert/sentiment-roberta-large-english --feature_segment roberta-sent --step_size 0.5 --device cuda
```

#### Using RoBERTa Model for Twitter Sentiment Analysis

```bash
python scripts/main.py --task perception --modality transcriptions --hf_model cardiffnlp/twitter-roberta-base-sentiment-latest --feature_segment roberta-twt-sentb --step_size 0.5 --device cuda
```

#### Using DistilRoBERTa Model for Emotion Detection

```bash
python scripts/main.py --task perception --modality transcriptions --hf_model j-hartmann/emotion-english-distilroberta-base --feature_segment distilroberta-emo --step_size 0.5 --device cuda
```

#### Using RoBERTa Model for GoEmotions Dataset

```bash
python scripts/main.py --task perception --modality transcriptions --hf_model SamLowe/roberta-base-go_emotions --feature_segment roberta-go-emo --step_size 0.5 --device cuda
```

### 6. Deactivate the Virtual Environment

After you're done, you can deactivate the virtual environment with the following command:

#### For `venv`

```bash
deactivate
```

#### For `conda`

```bash
conda deactivate
```

### Notes

- Ensure you have `python3` and `pip` installed on your system.
- If you encounter any issues with package installations, ensure that your `pip` is up to date by running `pip install --upgrade pip`.
