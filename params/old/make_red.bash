#!/usr/bin/env bash

set -euo pipefail

root=/Users/berquist/projects/aida/event-extraction
scriptdir=$root/repos/event-event-relations/cwc_event_event
reddir=$root/corpora/red_LDC2016T23
outdir=$root/corpora/red_LDC2016T23_flexnlp

python $scriptdir/ldcred.py flexnlp $reddir $outdir
