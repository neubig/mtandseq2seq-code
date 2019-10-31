This is an example of word alignment.

`model1.py` implements IBM model 1. You can run it by the following command:

    python model1.py ../en-de/train.en-de.low.filt.{de,en} | tee model1-probs.txt

The example of the output probabilities is in model-1-probs.txt.

We can also try some better models. For example, let's try [fast_align](https://github.com/clab/fast_align). First we install it. Then,

    paste ../en-de/train.en-de.low.filt.{de,en} | perl -p -e 's/\t/ ||| /g' > train.en-de.low.filt.deen
    fast_align -i train.en-de.low.filt.deen -d -o -v -p fastalign-probs.txt > fastalign-alignments.txt

Finally, let's visualize the outputs:

    perl visualize.pl ../en-de/train.en-de.low.filt.{de,en} fastalign-alignments.txt | less

fast_align works quite well on similar languages, but on very different languages (e.g. English and Chinese or Japanese), you often get better results by using GIZA++. GIZA++ can be used most easily through the [Moses](http://www.statmt.org/moses/) toolkit.
