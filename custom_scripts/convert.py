import torch
import os
from transformers import AutoConfig, AutoTokenizer, AutoModelForPreTraining

# model = torch.load('./BIOBERT_DIR/biobert_model.bin')

config = AutoConfig.from_json_file(
    '/Users/charangan/Desktop/Intern/NER/pretraining/bert/pretrained_models/bert-base-uncased/config.json')
model = AutoModelForPreTraining.from_pretrained('/Users/charangan/Desktop/Intern/NER/pretraining/bert/pretrained_models/uncased_L-12_H-768_A-12/bert_model.ckpt.index',
                                  from_tf=True,
                                  config=config)
tokenizer = AutoTokenizer.from_pretrained('/Users/charangan/Desktop/Intern/NER/pretraining/bert/pretrained_models/uncased_L-12_H-768_A-12/vocab.txt',
                                          do_lower_case=True)
# model.resize_token_embeddings(len(tokenizer))

model.save_pretrained(
    '/Users/charangan/Desktop/Intern/NER/pretraining/bert/pretrained_models/uncased_L-12_H-768_A-12/')
tokenizer.save_pretrained(
    '/Users/charangan/Desktop/Intern/NER/pretraining/bert/pretrained_models/uncased_L-12_H-768_A-12/')
