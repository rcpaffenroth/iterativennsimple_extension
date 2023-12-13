from transformers import AutoModelForSequenceClassification
#model = AutoModelForSequenceClassification.from_pretrained("t5-base")
#model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("mistralai/Mistral-7B-v0.1")
#model = AutoModelForSequenceClassification.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

print(model)

# print(model.bert)

# for x in model.modules():
#     print(x)