import torch

from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from accelerate import Accelerator

model_name="tiiuae/falcon-rw-1b"
dataset_name="yelp_review_full"
accelerator = Accelerator()

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
# Note, the pdding side is left since that is that the CausalLM model expects.
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
# NOTE: the tokenizer.pad_token is a special token that is used to pad sequences to the same length.
tokenizer.pad_token = tokenizer.eos_token

# NOT TESTED: I think this gets a batch of samples as defined by the map function.
# So, the longest refers to the longest sequence in the batch.
def tokenize_function(examples):
    return tokenizer(examples["text"], padding='longest', truncation=True, max_length=17)

# NOTE: the map function does some fancy caching.  I.e., the first time you run it, it will
# take a while.  But, the second time you run it, it will be much faster.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# We don't need the labels anymore, so we remove them.
tokenized_datasets = tokenized_datasets.remove_columns(["label", "text"])
# From https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes#datasets.Dataset.set_format
#     Set __getitem__ return format using this transform. The transform is applied on-the-fly on batches when __getitem__ is called. 
#     type (str, optional) — Either output type selected in [None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']. None means __getitem__ returns python objects (default).
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=4)

class Spy(torch.nn.Module):
        def __init__(self, model, debug=False):
            super().__init__()
            self.model = model
            self.debug = debug
            self.inputs = []
            self.outputs = []
            self.last_size = 0

        def reset(self):
            self.inputs = []
            self.outputs = []
            self.last_size = 0

        def forward(self, *args, **kwargs):
            self.inputs.append(args)
            output = self.model(*args, **kwargs)
            self.outputs.append(output)
            if self.debug:
                print(f'args {args}')
                print(f'kwargs {kwargs}')
                print(f'output {output}')
            return output

        def print_last_input(self):
            print(f'{self.last_size} {len(self.inputs)}')
            for i in range(self.last_size, len(self.inputs)):
                print(f'{i} {self.inputs[i][0].shape}')
            self.last_size = len(self.inputs)

model = AutoModelForCausalLM.from_pretrained(model_name)
my_spies = []

if model_name == "tiiuae/falcon-rw-1b":
    for i in range(len(model.transformer.h)):
        my_spy = Spy(model.transformer.h[i].self_attention)
        model.transformer.h[i].self_attention = my_spy
        my_spies.append(my_spy)
    my_spy = Spy(model.transformer.h[3].self_attention)
    model.transformer.h[3].self_attention = my_spy
# elif model_name == "bert-base-cased":
#     my_spy = Spy(model.bert.encoder.layer[3])
#     model.bert.encoder.layer[3] = my_spy
# elif model_name == "mistralai/Mistral-7B-v0.1":
#     my_spy = Spy(model.model.layers[5])
#     model.model.layers[5] = my_spy
else:
    raise ValueError(f"Unknown model {model_name}")

# Ok, now we do the same thing, but with a dataloader.
model, train_dataloader = accelerator.prepare(model, train_dataloader)

train_iterator = iter(train_dataloader)
for i in range(2):
    batch = next(train_iterator)
    input = batch['input_ids']

    output = model.generate(input, max_new_tokens=100, use_cache=False, do_sample=True, top_k=50, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
    print(my_spy.inputs[-1][0].shape)
    print(my_spy.outputs[-1][0].shape)

    for j in range(len(my_spies)):
        torch.save(my_spies[j].inputs[-1][0], f'inputs_{i}_{j}.pt')
        torch.save(my_spies[j].outputs[-1][0], f'outputs_{i}_{j}.pt')

    my_spy.reset()