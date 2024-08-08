# MuSe-2024 FealsGood


[Homepage](https://www.muse-challenge.org) || [Baseline Paper](#)


## Extracted Text Feature Segments 
[Google Drive](https://drive.google.com/drive/folders/1a4IPK_vag6mheRQCKWSEiAYISmCbT7DN?usp=sharing).

## Text Feature Extraction
[bert-b](https://huggingface.co/google-bert/bert-base-cased), [bert-l](https://huggingface.co/google-bert/bert-large-cased), [canine-c](https://huggingface.co/google/canine-c), [roberta-sent](https://huggingface.co/siebert/sentiment-roberta-large-english), [roberta-twt-sent](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), [distilroberta-emo](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base), [roberta-go-emo](https://huggingface.co/SamLowe/roberta-base-go_emotions)
  ```sh
  python scripts/main.py --task perception --modality transcriptions --hf_model google-bert/bert-base-cased --feature_segment bert-b --step_size 0.5 --device cuda
  python scripts/main.py --task perception --modality transcriptions --hf_model google-bert/bert-large-cased --feature_segment bert-l --step_size 0.5 --device cuda
  python scripts/main.py --task perception --modality transcriptions --hf_model google/canine-c --feature_segment canine-c --step_size 0.5 --device cuda
  python scripts/main.py --task perception --modality transcriptions --hf_model siebert/sentiment-roberta-large-english --feature_segment roberta-sent --step_size 0.5 --device cuda
  python scripts/main.py --task perception --modality transcriptions --hf_model cardiffnlp/twitter-roberta-base-sentiment-latest --feature_segment roberta-twt-sentb --step_size 0.5 --device cuda
  python scripts/main.py --task perception --modality transcriptions --hf_model j-hartmann/emotion-english-distilroberta-base --feature_segment distilroberta-emo --step_size 0.5 --device cuda
  python scripts/main.py --task perception --modality transcriptions --hf_model SamLowe/roberta-base-go_emotions --feature_segment roberta-go-emo --step_size 0.5 --device cuda
  ```