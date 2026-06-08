#!/usr/bin/env bash

mkdir data
cd data

curl -o gender_train.csv -L 'https://huggingface.co/datasets/dllllb/transactions-gender/resolve/main/gender_train.csv?download=true'
curl -o transactions.csv.gz -L 'https://huggingface.co/datasets/dllllb/transactions-gender/resolve/main/transactions.csv.gz?download=true'
curl -o tr_mcc_codes.csv -L 'https://huggingface.co/datasets/pytorch-lifestream/transactions-gender/resolve/main/tr_mcc_codes.csv?download=true'
curl -o tr_types.csv -L 'https://huggingface.co/datasets/pytorch-lifestream/transactions-gender/resolve/main/tr_types.csv?download=true'
gunzip -f *.csv.gz
