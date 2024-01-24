import torch

from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from datetime import datetime
from pathlib import Path
import json
from icecream import ic

from accelerate import Accelerator

from innsequence.model_splitter import model_splitter

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
    return tokenizer(examples["text"], padding='longest', max_length=64, truncation=True)

# NOTE: the map function does some fancy caching.  I.e., the first time you run it, it will
# take a while.  But, the second time you run it, it will be much faster.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# We don't need the labels anymore, so we remove them.
tokenized_datasets = tokenized_datasets.remove_columns(["label", "text"])
# From https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes#datasets.Dataset.set_format
#     Set __getitem__ return format using this transform. The transform is applied on-the-fly on batches when __getitem__ is called. 
#     type (str, optional) — Either output type selected in [None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']. None means __getitem__ returns python objects (default).
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=16)

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
my_spies = {}
my_spies_order = []

my_spies = model_splitter(model, model_name, wrapper=Spy)


# Ok, now we do the same thing, but with a dataloader.
model, train_dataloader = accelerator.prepare(model, train_dataloader)
train_iterator = iter(train_dataloader)

# Make a directory to store the results using pathlib with the current time and date
# as the name of the directory.
now = datetime.now()
current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
# the case of the path is my home directory.
save_path = Path.home() / Path(f'projects/inn_sequence/data/paths/{current_time}')
save_path.mkdir(parents=True, exist_ok=True)

# Save my_spies_order to a file.
with open(save_path / 'my_spies_order.json', 'w') as f:
    json.dump(my_spies_order, f)

model_path = save_path / 'model'
model_path.mkdir(parents=True, exist_ok=True)
for j, my_spy_name in enumerate(my_spies_order):
    try:
        torch.save(my_spies[my_spy_name].model, model_path / f'{my_spy_name}.pt')
    except AttributeError:
        print(f'Could not save {my_spy_name}')

for i in range(256):
    batch = next(train_iterator)
    input = batch['input_ids']
    print(f'input {input.shape}')

    print("Before model run - Memory Allocated: ", torch.cuda.memory_allocated()/10**9)
    print("Before model run - Memory Reserved:  ", torch.cuda.memory_reserved()/10**9)

    output = model(input)

    print("After model run - Memory Allocated:  ", torch.cuda.memory_allocated()/10**9)
    print("After model run - Memory Reserved:   ", torch.cuda.memory_reserved()/10**9)

    for j, my_spy_name in enumerate(my_spies.keys()):
        torch.save(my_spies[my_spy_name].inputs[-1][0], save_path / f'{my_spy_name}_inputs_batch_{i}.pt')
        torch.save(my_spies[my_spy_name].outputs[-1][0], save_path / f'{my_spy_name}_outputs_batch_{i}.pt')
        my_spies[my_spy_name].reset()
    torch.cuda.empty_cache()
