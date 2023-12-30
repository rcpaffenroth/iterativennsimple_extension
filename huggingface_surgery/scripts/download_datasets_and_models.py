
# %%
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from pathlib import Path

base_dir = Path('/mnt/research/rpaffenroth/data')
cache_dir = base_dir / 'huggingface_cache'
save_dir = base_dir / 'huggingface_downloads'

# data_download_list=['tiiuae/falcon-refinedweb', 'ptb_text_only']
data_download_list=['ptb_text_only']

# %%
for dataset_name in data_download_list:
    dataset = load_dataset(dataset_name, 
                            cache_dir=cache_dir)
    dataset.save_to_disk(save_dir / dataset_name)

# %%
# model_download_list=['tiiuae/falcon-rw-1b', 'tiiuae/falcon-7b', 'tiiuae/falcon-40b']
model_download_list=['tiiuae/falcon-rw-1b']

# %%
for model_name in model_download_list:
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                            cache_dir=cache_dir)
    model.save_pretrained(save_dir / model_name)
