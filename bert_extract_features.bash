#!/usr/bin/env bash

BASE_DIR=/Users/berquist/projects/aida/event-extraction
BERT_DIR=$BASE_DIR/repos/bert_github/bert
BERT_MODEL_DIR=$BASE_DIR/models/bert/uncased_L-12_H-768_A-12

name=stuvwx
INPUT_FILE="${BASE_DIR}/${name}.txt"
OUTPUT_FILE="${BASE_DIR}/${name}.jsonl"

# Sentence A and Sentence B are separated by the ||| delimiter for sentence
# pair tasks like question answering and entailment.
# For single sentence inputs, put one sentence per line and DON'T use the
# delimiter.

# echo 'Who was Jim Henson ? ||| Jim Henson was a puppeteer' > $INPUT_FILE
# printf "I'll have applesauce, please.\n" > $INPUT_FILE
printf "s t u v w x\n" > $INPUT_FILE
# printf "I'll have applesauce, please.\nYes, the regular kind.\n" > $INPUT_FILE

python $BERT_DIR/extract_features.py \
  --input_file=$INPUT_FILE \
  --output_file=$OUTPUT_FILE \
  --vocab_file=$BERT_MODEL_DIR/vocab.txt \
  --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
  --init_checkpoint=$BERT_MODEL_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=4
