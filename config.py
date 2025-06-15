import os
from huggingface_hub import hf_hub_download

token = os.getenv("HF_TOKEN")
model_repo = os.getenv("MODEL_REPO")
data_repo = os.getenv("DATA_REPO")

model_path=hf_hub_download(repo_id="model_repo", filename="FinBiLSTM.h5",repo_type="model",token=token)
config_path=hf_hub_download(repo_id="data_repo", filename="config.json",repo_type="dataset",token=token)
vocab_path=hf_hub_download(repo_id="data_repo", filename="vocab.npy",repo_type="dataset",token=token)