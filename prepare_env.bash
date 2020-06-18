#!/usr/bin/env bash

set -euo pipefail

project_base=/nas/gaia/users/berquist/event-extraction
conda create -n event-extraction-3.6 python=3.6
conda activate event-extraction-3.6
(
    cd ${project_base}/repos/isi-flexnlp
    pip install -e .
    pip install -r requirements.txt
    pip install -r requirements-optional.txt
)
(
    cd ${project_base}/repos/vistanlp-sandbox
    pip install -e .
    pip install -r requirements.txt
)
(
    cd ${project_base}/repos/gaia-event-extraction
    pip install -e .
    pip install -r requirements.txt
)
pip install tensorflow-gpu==1.14.0
python -m spacy download en
# Collecting en_core_web_sm==2.1.0
#   Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz (11.1MB)
#      |████████████████████████████████| 11.1MB 2.1MB/s
# Building wheels for collected packages: en-core-web-sm
#   Building wheel for en-core-web-sm (setup.py) ... done
#   Created wheel for en-core-web-sm: filename=en_core_web_sm-2.1.0-cp36-none-any.whl size=11074435 sha256=92ad7dc77bb4d1cc5bf2b562b97cc8203f5a56918d2b7c826d3e7315bd9d0ea1
#   Stored in directory: /tmp/pip-ephem-wheel-cache-x8s2q8nf/wheels/39/ea/3b/507f7df78be8631a7a3d7090962194cf55bc1158572c0be77f
# Successfully built en-core-web-sm
# Installing collected packages: en-core-web-sm
# Successfully installed en-core-web-sm-2.1.0
# ✔ Download and installation successful
# You can now load the model via spacy.load('en_core_web_sm')
# ✔ Linking successful
# /nas/home/berquist/opt/apps/python/miniconda3/envs/event-extraction-3.6/lib/python3.6/site-packages/en_core_web_sm
# -->
# /nas/home/berquist/opt/apps/python/miniconda3/envs/event-extraction-3.6/lib/python3.6/site-packages/spacy/data/en
# You can now load the model via spacy.load('en')

conda install -c pytorch cuda90
