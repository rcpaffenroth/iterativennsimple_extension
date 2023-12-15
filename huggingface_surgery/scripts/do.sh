#! /bin/bash

echo "train_and_analyze.py --model-name bert-base-cased --cuda"
python train_and_analyze.py --model-name bert-base-cased --cuda 
echo "train_and_analyze.py --model-name bert-base-cased"
python train_and_analyze.py --model-name bert-base-cased
echo "train_and_analyze.py --model-name bert-base-uncased --cuda" 
python train_and_analyze.py --model-name mistralai/Mistral-7B-v0.1 --cuda 
echo "train_and_analyze.py --model-name bert-base-uncased"
python train_and_analyze.py --model-name mistralai/Mistral-7B-v0.1 
