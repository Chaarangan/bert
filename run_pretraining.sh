#!/bin/bash

PRETRAINED_DIR=/Users/charangan/Desktop/Intern/NER/pretraining/bert/pretrained_models/Bio_ClinicalBERT

python run_pretraining.py \
  --input_file=./output/medbert.tfrecord \
  --output_dir=./output/medbert/ \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$PRETRAINED_DIR/config.json \
  --init_checkpoint=$PRETRAINED_DIR/model.ckpt-150000.index \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5