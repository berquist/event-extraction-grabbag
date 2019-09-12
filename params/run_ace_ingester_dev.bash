#!/usr/bin/env bash

set -euo pipefail

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python -m gaia_event_extraction.drivers.run_ace_or_ere_ingester \
       "${SCRIPTDIR}"/run_ace_ingester_dev.yaml
