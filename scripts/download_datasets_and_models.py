# %%
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from pathlib import Path
from subprocess import run

base_dir = Path('/scratch/rpaffenroth/data')
cache_dir = base_dir / 'huggingface_cache'
# save_dir = base_dir / 'huggingface_downloads'
home_cache_dir = Path('/home/rcpaffenroth/.cache/huggingface')

data_download_list=['tiiuae/falcon-refinedweb', 'yelp_review_full']
# data_download_list=['yelp_review_full']
model_download_list=['tiiuae/falcon-rw-1b', 'tiiuae/falcon-7b', 'tiiuae/falcon-40b']
# model_download_list=['tiiuae/falcon-rw-1b']

# run(['rm', '-rf', str(home_cache_dir)])
# run(['rm', '-rf', str(cache_dir)])

# %%
for dataset_name in data_download_list:

    run(['rcp', 'notify', f'start {dataset_name}'])
    dataset = load_dataset(dataset_name, 
                            cache_dir=cache_dir)
    # dataset.save_to_disk(save_dir / dataset_name)
    print(f'dataset try 1 {dataset_name}')
    run(['du', '-sh', str(home_cache_dir)])
    run(['du', '-sh', str(cache_dir)])

    run(['rcp', 'notify', f'middle {dataset_name}'])
    dataset = load_dataset(dataset_name, 
                            cache_dir=cache_dir)
    # dataset.save_to_disk(save_dir / dataset_name)
    print(f'dataset try 2 {dataset_name}')
    run(['du', '-sh', str(home_cache_dir)])
    run(['du', '-sh', str(cache_dir)])

    run(['rcp', 'notify', f'end {dataset_name}'])

# %%
for model_name in model_download_list:

    run(['rcp', 'notify', f'start {model_name}'])
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                            cache_dir=cache_dir)
    print(f'model try 1 {model_name}')
    # model.save_pretrained(save_dir / model_name)
    run(['du', '-sh', str(home_cache_dir)])
    run(['du', '-sh', str(cache_dir)])

    run(['rcp', 'notify', f'middle {model_name}'])
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                            cache_dir=cache_dir)
    print(f'model try 2 {model_name}')
    # model.save_pretrained(save_dir / model_name)
    run(['du', '-sh', str(home_cache_dir)])
    run(['du', '-sh', str(cache_dir)])

    run(['rcp', 'notify', f'end {model_name}'])
