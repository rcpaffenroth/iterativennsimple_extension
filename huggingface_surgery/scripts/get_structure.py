from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM
from transformers import AutoConfig
#model = AutoModelForSequenceClassification.from_pretrained("t5-base")
#model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
#model = AutoModelForSequenceClassification.from_pretrained("mistralai/Mistral-7B-v0.1")
#model = AutoModelForSequenceClassification.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
# model = AutoModelForSequenceClassification.from_pretrained("tiiuae/falcon-rw-1b")

# print(model)

# model = AutoConfig.from_pretrained("tiiuae/falcon-40b")
# print(model)

model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b")
print(model)

# model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b")
# print(model)

# model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
# print(model)

