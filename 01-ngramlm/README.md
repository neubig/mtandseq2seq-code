# Interpolated bi-gram model code example
by Graham Neubig

This is an example of an interpolated bi-gram language model.

## Basic Usage

Usage:

    python ngram_lm.py train_data.txt test_data.txt

For example, on the example data in the top directory:

    python ngram_lm.py ../en-de/train.en-de.low.en ../en-de/valid.en-de.low.en

You can also set some hyper-parameters

    python ngram_lm.py --uni_prob 0.3 --unk_prob 0.005 ../en-de/train.en-de.low.en ../en-de/valid.en-de.low.en

## Advanced Examples

You can perform grid search to find the best interpolation coefficients, an example is shown in `grid_search.sh`

    bash grid_search.sh

You can also print out the probabilities of each word:

    python ngram_lm.py --print_probs ../en-de/train.en-de.low.en ../en-de/valid.en-de.low.en > result-probs.txt

You can also visualize the probabilities:

    python visualize_probs.py < result-probs.txt > result-probs.html
