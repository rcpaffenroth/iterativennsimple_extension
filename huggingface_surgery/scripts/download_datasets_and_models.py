
# %%
from datasets import load_dataset
from transformers import AutoModelForCausalLM

data_download_list=['tiiuae/falcon-refinedweb', 'ptb_text_only']
# data_download_list=['ptb_text_only']

# %%
for dataset_name in data_download_list:
    dataset = load_dataset(dataset_name, cache_dir='/home/rcpaffenroth/projects/quick_project/huggingface_surgery/huggingface_cache')
    dataset.save_to_disk('/home/rcpaffenroth/projects/quick_project/huggingface_surgery/huggingface_downloads/'+dataset_name)

# # %%
# for dataset_name in data_download_list:
#     dataset = load_dataset('/home/rcpaffenroth/projects/quick_project/huggingface_surgery/huggingface_downloads/'+dataset_name)

# %%
model_download_list=['tiiuae/falcon-rw-1b', 'tiiuae/falcon-7b', 'tiiuae/falcon-40b']
# model_download_list=['tiiuae/falcon-rw-1b']

# %%
for model_name in model_download_list:
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='/home/rcpaffenroth/projects/quick_project/huggingface_surgery/huggingface_cache')
    model.save_pretrained('/home/rcpaffenroth/projects/quick_project/huggingface_surgery/huggingface_downloads/'+model_name)

# # %%
# for model_name in model_download_list:
#     model = AutoModelForCausalLM.from_pretrained('/home/rcpaffenroth/projects/quick_project/huggingface_surgery/huggingface_downloads/'+model_name)


# %%
