#!/usr/bin/env bash

set -euo pipefail

# SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python -m gaia_event_extraction.drivers.score_model_for_ace_data \
       /Users/berquist/projects/aida/event-extraction/repos/gaia-event-extraction/sample_params/neural_trigger_model_ace/04_score_model_for_ace_data_train.yaml
       # "${SCRIPTDIR}"/score_model_for_ace_data_train.yaml
