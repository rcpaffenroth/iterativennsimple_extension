#! /bin/bash

COMMAND="train_and_analyze.py --model-name bert-base-cased --cuda"
rcp n "$COMMAND started"
python $COMMAND 
rcp n "$COMMAND ended"

COMMAND="train_and_analyze.py --model-name bert-base-cased"
rcp n "$COMMAND started"
python $COMMAND 
rcp n "$COMMAND ended"

COMMAND="train_and_analyze.py --model-name bert-base-uncased --cuda" 
rcp n "$COMMAND started"
python $COMMAND 
rcp n "$COMMAND ended"

COMMAND="train_and_analyze.py --model-name bert-base-uncased"
rcp n "$COMMAND started"
python $COMMAND 
rcp n "$COMMAND ended"
