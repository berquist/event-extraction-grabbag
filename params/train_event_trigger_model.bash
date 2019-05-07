#!/usr/bin/env bash

python -m gaia_event_extraction.model_trainers.train_event_trigger_model \
       params_ace_single_split_trigger_model.yaml
