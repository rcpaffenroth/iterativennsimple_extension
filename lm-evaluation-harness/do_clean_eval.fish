#! /usr/bin/env fish

set install false

if test $install = true
    rm -rf test
    mkdir test
    cd test
    python3 -m venv venv
    source venv/bin/activate.fish
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git
    cd lm-evaluation-harness
    # version from https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
    git checkout b281b09
    # revision from https://huggingface.co/datasets/open-llm-leaderboard/details_tiiuae__falcon-rw-1b/blob/main/results_2023-10-25T18-16-05.784566.json
    pip install .
else
    cd test
    source venv/bin/activate.fish
    cd lm-evaluation-harness
end

#python main.py --model=hf-causal-experimental --model_args="pretrained=tiiuae/falcon-rw-1b,revision=e4b9872bb803165eb22f0a867d4e6a64d34fce19,use_accelerate=True" --tasks=winogrande --num_fewshot=5 --batch_size=8 --output_path=../output

python main.py --model=hf-causal-experimental --model_args="pretrained=tiiuae/falcon-rw-1b,revision=e4b9872bb803165eb22f0a867d4e6a64d34fce19,use_accelerate=True" --tasks=drop --num_fewshot=3 --batch_size=8 --output_path=../output
