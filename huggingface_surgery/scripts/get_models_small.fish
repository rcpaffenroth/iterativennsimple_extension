#! /usr/bin/env fish

# set -l model_download_list tiiuae/falcon-rw-1b tiiuae/falcon-7b tiiuae/falcon-40b
set -l model_download_list tiiuae/falcon-rw-1b 
# set -l data_download_list tiiuae/falcon-refinedweb ptb_text_only
set -l data_download_list ptb_text_only

set -l cache_dir /mnt/research/rpaffenroth/data/huggingface-cache

for model in $model_download_list
    message "Downloading $model"
    huggingface-cli download --cache-dir $cache_dir $model
    message "Finished downloading $model"
end

for data in $data_download_list
    message "Downloading $data"
    huggingface-cli download --repo-type dataset --cache-dir $cache_dir $data
    message "Finished downloading $data"
end
