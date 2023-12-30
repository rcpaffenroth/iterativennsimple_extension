from datasets import load_dataset

cache_dir = "/scratch/rcpaffenroth/data/huggingface-cache"

data = load_dataset("ptb_text_only", cache_dir=cache_dir)
print(data)
