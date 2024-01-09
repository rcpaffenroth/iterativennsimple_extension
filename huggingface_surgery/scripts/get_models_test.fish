#! /usr/bin/env fish

. ../.venv/bin/activate.fish

# set -l model_download_list tiiuae/falcon-rw-1b tiiuae/falcon-7b tiiuae/falcon-40b
set -l model_download_list tiiuae/falcon-rw-1b 
# set -l data_download_list tiiuae/falcon-refinedweb yelp_review_full
set -l data_download_list yelp_review_full

set -l cache_dir /scratch/rcpaffenroth/data/huggingface-cache

rm -rf ~/.cache/huggingface/**
rm -rf $cache_dir/**

du -sh ~/.cache/huggingface
du -sh $cache_dir

for model in $model_download_list
    message "Downloading $model"
    huggingface-cli download --revision main --cache-dir $cache_dir $model
    message "Finished downloading $model"
end

for data in $data_download_list
    message "Downloading $data"
    huggingface-cli download --revision main --repo-type dataset --cache-dir $cache_dir $data
    message "Finished downloading $data"
end

du -sh ~/.cache/huggingface
du -sh $cache_dir

python get_structure.py

du -sh ~/.cache/huggingface
du -sh $cache_dir

python get_data.py

du -sh ~/.cache/huggingface
du -sh $cache_dir
