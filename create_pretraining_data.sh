#!/bin/bash

python create_pretraining_data.py \
  --input_file=/Users/charangan/Desktop/Intern/NER/pretraining/custom/data/clinical_wiki.txt \
  --output_file=./output/medbert.tfrecord \
  --vocab_file=/Users/charangan/Desktop/Intern/NER/pretraining/bert/pretrained_models/Bio_ClinicalBERT/vocab.txt \
  --do_lower_case=False \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5