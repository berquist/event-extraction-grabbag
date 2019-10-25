#!/usr/bin/env bash

set -euo pipefail

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python -m gaia_event_extraction.drivers.run_bert_annotator_on_ace \
       "${SCRIPTDIR}"/../repos/gaia-event-extraction/sample_params/run_bert_annotator_on_ace_train.params
