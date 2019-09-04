# Neural encoder decoder model code example
by Antonis Anastasopoulos

This is an example of a simple encoder decoder model, based on the DyNet examples.

## Data

We will use a different enc-dec example: morphological inflection.
The task is given a sequence of input morphological tags and the lemma of a word, to produce the inflected form.
The Asturian data (taken from the SIGMORPHON 2019 challenge) provide the input and the output in each line, separated by `|||`.
	

## Basic Usage

	py encdec.py [train] [dev] [test]

