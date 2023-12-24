#! /usr/bin/env fish

set -l download_list tiiuae/falcon-rw-1b tiiuae/falcon-7b tiiuae/falcon-40b tiiuae/falcon-refinedweb
# set -l download_list tiiuae/falcon-rw-1b 

cd huggingface_downloads
for model in $download_list
    message "Downloading $model"
    git clone https://huggingface.co/$model
    message "Finished downloading $model"
end
