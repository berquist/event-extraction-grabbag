#!/usr/bin/env bash

set -euo pipefail

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python -m gaia_event_extraction.drivers.run_model_for_ace_data \
       "${SCRIPTDIR}"/run_model_for_ace_data_test.yaml
