from transformers import AutoModelForCausalLM

cache_dir = "/mnt/research/rpaffenroth/data/huggingface-cache"

model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b", cache_dir=cache_dir)
print(model)

# model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", cache_dir=cache_dir)
# print(model)

# model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b", cache_dir=cache_dir)
# print(model)

