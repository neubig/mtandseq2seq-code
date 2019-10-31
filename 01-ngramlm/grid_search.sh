#!/bin/bash

for uni_alpha in $(seq 0.01 0.01 0.5); do
  echo "--- uni_alpha=$uni_alpha"
  python ngram_lm.py --uni_alpha $uni_alpha ../en-de/train.en-de.low.en ../en-de/valid.en-de.low.en
done
