import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import get_scheduler
from transformers import AutoTokenizer
from datasets import load_dataset
import evaluate

dataset = load_dataset("yelp_review_full")
dataset["train"][100]

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

class Spy(torch.nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.model = model
        self.debug = debug
        self.inputs = []
        self.outputs = []

    def forward(self, *args, **kwargs):
        self.inputs.append(args)
        output = self.model(*args, **kwargs)
        self.outputs.append(output)
        if self.debug:
            print(f'args {args}')
            print(f'kwargs {kwargs}')
            print(f'output {output}')
        return output

from transformers import AutoModelForSequenceClassification
#model_name = "bert-base-cased"
model_name = "mistralai/Mistral-7B-v0.1"
# The num+labels=5 means that the final classification layer will have 5 outputs
# and will be unitialized, since the number of labels is not known.
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
print(model)

if model_name == "bert-base-cased":
    my_spy = Spy(model.bert.encoder.layer[3])
    model.bert.encoder.layer[3] = my_spy
elif model_name == "mistralai/Mistral-7B-v0.1":
    my_spy = Spy(modelr.layers[5])
    model.layers[5] = my_spy
else:
    raise ValueError(f"Unknown model {model_name}")

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())
