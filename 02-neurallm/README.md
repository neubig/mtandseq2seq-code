# Interpolated bi-gram model code example
by Antonis Anastasopoulos

This is an example of a simple LSTM neural language model.

## Preparing the Data
	
First we find the word with frequency larger than 1

	py replace_unk.py ../en-de/train.en-de.low.en ../en-de/train.en-de.low.unk.en 1 ../en-de/vocab.en

And make sure that UNKs are also replaced with `<UNK>` in the dev and test

	py replace_unk_given_vocab.py ../en-de/valid.en-de.low.en ../en-de/valid.en-de.low.unk.en ../en-de/vocab.en
	py replace_unk_given_vocab.py ../en-de/test.en-de.low.en ../en-de/test.en-de.low.unk.en ../en-de/vocab.en

## Basic Usage

	py rnnlm.py train dev test --train

Use `rnnlm.py` for a character-level LM on the example data in the top directory:

    py rnnlm.py ../en-de/train.en-de.low.unk.en  ../en-de/valid.en-de.low.unk.en ../en-de/test.en-de.low.unk.en --dynet-autobatch 1 --dynet-mem 4000

## Advanced Examples

You can also print out the probabilities of each word:

    python ngram_lm.py --print_probs ../en-de/train.en-de.low.en ../en-de/valid.en-de.low.en > result-probs.txt
