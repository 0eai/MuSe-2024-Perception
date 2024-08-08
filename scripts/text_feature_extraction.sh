# Navigate to the parent directory
cd..

# Clone the MultimodalFeatureExtractor repository from GitHub
git clone https://github.com/0eai/MultimodalFeatureExtractor.git

# Change directory to the cloned repository
cd MultimodalFeatureExtractor

# Create a new virtual environment named 'multimodal_env'
python -m venv multimodal_env

# Activate the virtual environment
# For Windows
# .\multimodal_env\Scripts\activate

# For macOS and Linux
source multimodal_env/bin/activate

# Install the required Python packages specified in requirements.txt
pip install -r requirements.txt

# Execute the main script for the perception task using different models and configurations

# Using BERT base model
python scripts/main.py --task perception --modality transcriptions --hf_model google-bert/bert-base-cased --feature_segment bert-b --step_size 0.5 --device cuda

# Using BERT large model
python scripts/main.py --task perception --modality transcriptions --hf_model google-bert/bert-large-cased --feature_segment bert-l --step_size 0.5 --device cuda

# Using CANINE model
python scripts/main.py --task perception --modality transcriptions --hf_model google/canine-c --feature_segment canine-c --step_size 0.5 --device cuda

# Using RoBERTa model for sentiment analysis
python scripts/main.py --task perception --modality transcriptions --hf_model siebert/sentiment-roberta-large-english --feature_segment roberta-sent --step_size 0.5 --device cuda

# Using RoBERTa model for Twitter sentiment analysis
python scripts/main.py --task perception --modality transcriptions --hf_model cardiffnlp/twitter-roberta-base-sentiment-latest --feature_segment roberta-twt-sentb --step_size 0.5 --device cuda

# Using DistilRoBERTa model for emotion detection
python scripts/main.py --task perception --modality transcriptions --hf_model j-hartmann/emotion-english-distilroberta-base --feature_segment distilroberta-emo --step_size 0.5 --device cuda

# Using RoBERTa model for GoEmotions dataset
python scripts/main.py --task perception --modality transcriptions --hf_model SamLowe/roberta-base-go_emotions --feature_segment roberta-go-emo --step_size 0.5 --device cuda
