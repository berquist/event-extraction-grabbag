#!/usr/bin/env bash

BERT_BASE_DIR=/Users/berquist/projects/aida/event-extraction/models/bert/uncased_L-12_H-768_A-12/
SQUAD_DIR=/Users/berquist/projects/aida/event-extraction/squad_data

# SQuAD 1.1
# python run_squad.py \
#   --vocab_file=$BERT_BASE_DIR/vocab.txt \
#   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
#   --do_train=True \
#   --train_file=$SQUAD_DIR/train-v1.1.json \
#   --do_predict=True \
#   --predict_file=$SQUAD_DIR/dev-v1.1.json \
#   --train_batch_size=12 \
#   --learning_rate=3e-5 \
#   --num_train_epochs=2.0 \
#   --max_seq_length=384 \
#   --doc_stride=128 \
#   --output_dir=/tmp/squad_base/

# SQuAD 2.0
# python run_squad.py \
#   --vocab_file=$BERT_LARGE_DIR/vocab.txt \
#   --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
#   --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
#   --do_train=True \
#   --train_file=$SQUAD_DIR/train-v2.0.json \
#   --do_predict=True \
#   --predict_file=$SQUAD_DIR/dev-v2.0.json \
#   --train_batch_size=24 \
#   --learning_rate=3e-5 \
#   --num_train_epochs=2.0 \
#   --max_seq_length=384 \
#   --doc_stride=128 \
#   --output_dir=gs://some_bucket/squad_large/ \
#   --use_tpu=True \
#   --tpu_name=$TPU_NAME \
#   --version_2_with_negative=True
python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=False \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=False \
  --version_2_with_negative=True
