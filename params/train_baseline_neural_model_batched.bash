#!/usr/bin/env bash

set -euo pipefail

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python -m gaia_event_extraction.model_trainers.train_baseline_neural_model_batched \
       "${SCRIPTDIR}"/train_baseline_neural_model_batched.yaml
