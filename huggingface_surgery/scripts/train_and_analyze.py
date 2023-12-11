# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# Based on https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/pytorch/training.ipynb

# %% [markdown] id="tnY-H8AHJg-p"
# # Fine-tune a pretrained model

# %% [markdown] id="edsVlcP2Jg-q"
# There are significant benefits to using a pretrained model. It reduces computation costs, your carbon footprint, and allows you to use state-of-the-art models without having to train one from scratch. 🤗 Transformers provides access to thousands of pretrained models for a wide range of tasks. When you use a pretrained model, you train it on a dataset specific to your task. This is known as fine-tuning, an incredibly powerful training technique. In this tutorial, you will fine-tune a pretrained model with a deep learning framework of your choice:
#
# * Fine-tune a pretrained model with 🤗 Transformers [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer).
# * Fine-tune a pretrained model in TensorFlow with Keras.
# * Fine-tune a pretrained model in native PyTorch.
#
# <a id='data-processing'></a>

# %%
import pyarrow
pyarrow.__version__

# %% [markdown] id="T_nFSgR4Jg-r"
# ## Prepare a dataset

# %% [markdown] id="YfvAqJ4zJg-s"
# Before you can fine-tune a pretrained model, download a dataset and prepare it for training. The previous tutorial showed you how to process data for training, and now you get an opportunity to put those skills to the test!
#
# Begin by loading the [Yelp Reviews](https://huggingface.co/datasets/yelp_review_full) dataset:

# %% id="7RdtDYgwJg-s" outputId="672bedb5-a44d-45cf-ddd1-3b754e47d419"
from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
dataset["train"][100]

# %% [markdown] id="XIZowxoFJg-t"
# As you now know, you need a tokenizer to process the text and include a padding and truncation strategy to handle any variable sequence lengths. To process your dataset in one step, use 🤗 Datasets [`map`](https://huggingface.co/docs/datasets/process.html#map) method to apply a preprocessing function over the entire dataset:

# %% id="vzPnjiPkJg-t"
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# %% [markdown] id="o2vZH5zlJg-t"
# If you like, you can create a smaller subset of the full dataset to fine-tune on to reduce the time it takes:

# %% id="my9J-SuxJg-u"
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# %% [markdown] id="rzRI3IszJg-u"
# <a id='trainer'></a>

# %% [markdown] id="B6CINlZ4Jg-u"
# ## Train

# %% [markdown] id="gLiuq85jJg-u"
# At this point, you should follow the section corresponding to the framework you want to use. You can use the links
# in the right sidebar to jump to the one you want - and if you want to hide all of the content for a given framework,
# just use the button at the top-right of that framework's block!

# %% [markdown] id="siQ-0alhJg-u"
# ## Train with PyTorch Trainer

# %% [markdown] id="Yw0EYIJRJg-v"
# 🤗 Transformers provides a [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) class optimized for training 🤗 Transformers models, making it easier to start training without manually writing your own training loop. The [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) API supports a wide range of training options and features such as logging, gradient accumulation, and mixed precision.
#
# Start by loading your model and specify the number of expected labels. From the Yelp Review [dataset card](https://huggingface.co/datasets/yelp_review_full#data-fields), you know there are five labels:

# %% id="fRyoN1zyJg-v"
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

# %%
model

# %%
import torch

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


# %%
my_spy = Spy(model.bert.encoder.layer[3])
model.bert.encoder.layer[3] = my_spy

# %% [markdown] id="Frb_ytB1Jg-v"
# <Tip>
#
# You will see a warning about some of the pretrained weights not being used and some weights being randomly
# initialized. Don't worry, this is completely normal! The pretrained head of the BERT model is discarded, and replaced with a randomly initialized classification head. You will fine-tune this new model head on your sequence classification task, transferring the knowledge of the pretrained model to it.
#
# </Tip>

# %% [markdown] id="lQPsIdt2Jg-v"
# ### Training hyperparameters

# %% [markdown] id="vDLASz-5Jg-v"
# Next, create a [TrainingArguments](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) class which contains all the hyperparameters you can tune as well as flags for activating different training options. For this tutorial you can start with the default training [hyperparameters](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments), but feel free to experiment with these to find your optimal settings.
#
# Specify where to save the checkpoints from your training:

# %% id="GCpEVdTEJg-v"
from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")

# %% [markdown] id="acUbdXP-Jg-v"
# ### Evaluate

# %% [markdown] id="Ho4_nwUOJg-v"
# [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) does not automatically evaluate model performance during training. You'll need to pass [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) a function to compute and report metrics. The [🤗 Evaluate](https://huggingface.co/docs/evaluate/index) library provides a simple [`accuracy`](https://huggingface.co/spaces/evaluate-metric/accuracy) function you can load with the [evaluate.load](https://huggingface.co/docs/evaluate/main/en/package_reference/loading_methods#evaluate.load) (see this [quicktour](https://huggingface.co/docs/evaluate/a_quick_tour) for more information) function:

# %% id="Y8x9OsB8Jg-w"
import numpy as np
import evaluate

metric = evaluate.load("accuracy")


# %% [markdown] id="amhvknUhJg-w"
# Call `compute` on `metric` to calculate the accuracy of your predictions. Before passing your predictions to `compute`, you need to convert the predictions to logits (remember all 🤗 Transformers models return logits):

# %% id="1V_PbTOzJg-w"
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# %% [markdown] id="W80YVxcdJg-w"
# If you'd like to monitor your evaluation metrics during fine-tuning, specify the `evaluation_strategy` parameter in your training arguments to report the evaluation metric at the end of each epoch:

# %% id="CEuInffoJg-w"
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

# %% [markdown] id="2Y7rtzHKJg-w"
# ### Trainer

# %% [markdown] id="oLZZGfYlJg-w"
# Create a [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) object with your model, training arguments, training and test datasets, and evaluation function:

# %% id="hZGPGh3gJg-w"
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# %% [markdown] id="lgCd-y0DJg-x"
# Then fine-tune your model by calling [train()](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.train):

# %% id="wVcjSvH9Jg-x"
trainer.train()

# %%
print(len(my_spy.inputs[0]))
print(len(my_spy.outputs[0]))


# %%
print(my_spy.inputs[0][1].shape)
print(my_spy.inputs[0][1])

# %% [markdown] id="z2DJKS9qJg-x"
# <a id='pytorch_native'></a>

# %% [markdown] id="K58YX3EDJg-x"
# ## Train in native PyTorch

# %% [markdown] id="VfUPHegVJg-x"
# [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) takes care of the training loop and allows you to fine-tune a model in a single line of code. For users who prefer to write their own training loop, you can also fine-tune a 🤗 Transformers model in native PyTorch.
#
# At this point, you may need to restart your notebook or execute the following code to free some memory:

# %%
import torch

# %% id="QYpIsHiJJg-x"
del model
del trainer
torch.cuda.empty_cache()

# %% [markdown] id="udetg3l9Jg-2"
# Next, manually postprocess `tokenized_dataset` to prepare it for training.
#
# 1. Remove the `text` column because the model does not accept raw text as an input:
#
#     ```py
#     >>> tokenized_datasets = tokenized_datasets.remove_columns(["text"])
#     ```
#
# 2. Rename the `label` column to `labels` because the model expects the argument to be named `labels`:
#
#     ```py
#     >>> tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
#     ```
#
# 3. Set the format of the dataset to return PyTorch tensors instead of lists:
#
#     ```py
#     >>> tokenized_datasets.set_format("torch")
#     ```
#
# Then create a smaller subset of the dataset as previously shown to speed up the fine-tuning:

# %%
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# %% id="dsXtVjbiJg-2"
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# %% [markdown] id="irLG3JZ0Jg-2"
# ### DataLoader

# %% [markdown] id="dBD_WQe_Jg-2"
# Create a `DataLoader` for your training and test datasets so you can iterate over batches of data:

# %% id="bNvBk2nGJg-2"
from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

# %% [markdown] id="EZXtfpDJJg-3"
# Load your model with the number of expected labels:

# %% id="n56UlSS7Jg-3"
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

# %% [markdown] id="XCONNhpfJg-3"
# ### Optimizer and learning rate scheduler

# %% [markdown] id="VKfs1fWHJg-3"
# Create an optimizer and learning rate scheduler to fine-tune the model. Let's use the [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) optimizer from PyTorch:

# %% id="eU7OYJIBJg-3"
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

# %% [markdown] id="66tEFIU9Jg-3"
# Create the default learning rate scheduler from [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer):

# %% id="vorgq4b1Jg-3"
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# %% [markdown] id="aMWYQU4-Jg-4"
# Lastly, specify `device` to use a GPU if you have access to one. Otherwise, training on a CPU may take several hours instead of a couple of minutes.

# %% id="uNmsFNNFJg-4"
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# %% [markdown] id="L6jUCZOOJg-4"
# <Tip>
#
# Get free access to a cloud GPU if you don't have one with a hosted notebook like [Colaboratory](https://colab.research.google.com/) or [SageMaker StudioLab](https://studiolab.sagemaker.aws/).
#
# </Tip>
#
# Great, now you are ready to train! 🥳

# %% [markdown] id="WO2X_Nf3Jg-4"
# ### Training loop

# %% [markdown] id="AwO8ZbwPJg-4"
# To keep track of your training progress, use the [tqdm](https://tqdm.github.io/) library to add a progress bar over the number of training steps:

# %% id="z-1RoKvgJg-4"
from tqdm.auto import tqdm

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

# %% [markdown] id="NY7jJjJGJg-4"
# ### Evaluate

# %% [markdown] id="gpEhNxrAJg-5"
# Just like how you added an evaluation function to [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer), you need to do the same when you write your own training loop. But instead of calculating and reporting the metric at the end of each epoch, this time you'll accumulate all the batches with `add_batch` and calculate the metric at the very end.

# %% id="Ip4iy2MJJg-5"
import evaluate

metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()

# %% [markdown] id="rdQ7rJUKJg-5"
# <a id='additional-resources'></a>

# %% [markdown] id="I-Xklnj3Jg-5"
# ## Additional resources

# %% [markdown] id="nlKLalqAJg-5"
# For more fine-tuning examples, refer to:
#
# - [🤗 Transformers Examples](https://github.com/huggingface/transformers/tree/main/examples) includes scripts
#   to train common NLP tasks in PyTorch and TensorFlow.
#
# - [🤗 Transformers Notebooks](https://huggingface.co/docs/transformers/main/en/notebooks) contains various notebooks on how to fine-tune a model for specific tasks in PyTorch and TensorFlow.
