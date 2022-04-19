#! /bin/bash
source activate ./envs
python trainer.py fit --config config/base.yaml --config  config/classification.yaml