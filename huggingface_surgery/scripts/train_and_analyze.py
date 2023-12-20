import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import get_scheduler
from transformers import AutoTokenizer
from datasets import load_dataset
import evaluate

from transformers import AutoModelForCausalLM

import click

def run(model_name, cuda, fine_tune, dataset_name="yelp_review_full"):
    # Load the dataset.  Just load the train split.  Different datasets have different splits, but
    # having a train split is common.
    dataset = load_dataset(dataset_name, split="train")

    # We only want to use a small subset of the dataset for this example.
    # Note that dataset[:1000] would seem to work, but it doesn't.  In particular,
    # dataset[:1000] is not a dataset object, but a dictionary.  So, we use the
    # select method to get a dataset object.
    dataset = dataset.shuffle(42).select(range(1000))

    # This converts the dataset to a format that the model can understand.
    # In particlar, it takes the words and converts them to numbers/tokens.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # NOTE: the tokenizer.pad_token is a special token that is used to pad sequences to the same length.
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # NOTE: the map function does some fancy caching.  I.e., the first time you run it, it will
    # take a while.  But, the second time you run it, it will be much faster.
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # We don't need the labels anymore, so we remove them.
    tokenized_datasets = tokenized_datasets.remove_columns(["label", "text"])
    # From https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes#datasets.Dataset.set_format
    #     Set __getitem__ return format using this transform. The transform is applied on-the-fly on batches when __getitem__ is called. 
    #     type (str, optional) — Either output type selected in [None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']. None means __getitem__ returns python objects (default).
    tokenized_datasets.set_format("torch")

    # A little sanity check.
    print('example text')
    print(dataset[0])
    print('example tokenized text')
    print(tokenized_datasets[0])
    print('example decoded tokenized text')
    print(tokenizer.decode(tokenized_datasets[0]['input_ids'][:10]))

    train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=8)

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

    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(model)

    if model_name == "tiiuae/falcon-rw-1b":
        my_spy = Spy(model.transformer.h[3])
        model.transformer.h[3] = my_spy
    elif model_name == "bert-base-cased":
        my_spy = Spy(model.bert.encoder.layer[3])
        model.bert.encoder.layer[3] = my_spy
    elif model_name == "mistralai/Mistral-7B-v0.1":
        my_spy = Spy(model.model.layers[5])
        model.model.layers[5] = my_spy
    else:
        raise ValueError(f"Unknown model {model_name}")
        
    if cuda:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f'device {device}')
    model.to(device)

    prompt = "This is a review of a restaurant.  The food was"
    input = tokenizer(prompt, return_tensors="pt").input_ids
    input.to(device)
    print(input)

    output = model.generate(input, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95, pad_token_id=tokenizer.pad_token_id)
    generated_text = tokenizer.decode(output[0])

    print(f'prompt {prompt}')
    print(f'generated_text {generated_text}')

    # input = tokenized_datasets['input_ids'][:10]
    # output = model.generate(input, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

    # for i in range(len(input)):
    #     print(f'input {tokenizer.decode(input[i])}')
    #     print(f'output {tokenizer.decode(output[i])}')

    # if fine_tune:
    #     optimizer = AdamW(model.parameters(), lr=5e-5)

    #     num_epochs = 3
    #     num_training_steps = num_epochs * len(train_dataloader)
    #     lr_scheduler = get_scheduler(
    #         name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    #     )

    #     progress_bar = tqdm(range(num_training_steps))

    #     model.train()
    #     for epoch in range(num_epochs):
    #         for batch in train_dataloader:
    #             batch = {k: v.to(device) for k, v in batch.items()}

    #             outputs = model(**batch)
    #             loss = outputs.loss
        #             loss.backward()

    #             optimizer.step()
    #             lr_scheduler.step()
    #             optimizer.zero_grad()
    #             progress_bar.update(1)

    # metric = evaluate.load("accuracy")
    # model.eval()
    # for batch in eval_dataloader:
    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     with torch.no_grad():
    #         outputs = model(**batch)

    #     logits = outputs.logits
    #     predictions = torch.argmax(logits, dim=-1)
    #     metric.add_batch(predictions=predictions, references=batch["labels"])

    # print(metric.compute())

@click.command()
@click.option('--model-name', default="tiiuae/falcon-rw-1b")
@click.option('--cuda', is_flag=True)
@click.option('--fine-tune', is_flag=True)
def cli(model_name, cuda, fine_tune):
    run(model_name, cuda, fine_tune)

if __name__ == "__main__":
    cli()