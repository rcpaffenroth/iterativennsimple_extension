from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b", cache_dir="/home/rcpaffenroth/projects/quick_project/huggingface_surgery/cache")
print(model)

model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", cache_dir="/home/rcpaffenroth/projects/quick_project/huggingface_surgery/cache")
print(model)

# model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b", cache_dir="/home/rcpaffenroth/projects/quick_project/huggingface_surgery/cache")
# print(model)

