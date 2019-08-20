# --- Interpolated bi-gram model code example
# by Graham Neubig

import sys
import math
from collections import defaultdict

N = 2
VOCAB_SIZE = 10000000
UNK_ALPHA = 0.01
UNI_ALPHA = 0.25
BI_ALPHA = 1.0 - UNK_ALPHA - UNI_ALPHA

train_counts = defaultdict(lambda: 0)
train_ctxts = defaultdict(lambda: 0)
with open(sys.argv[1], "r") as f:
  for line in f:
    sent = line.strip().split(" ") + ["<s>"]
    ngram = ["<s>"] * N
    for word in sent:
      ctxt = ngram[1:]
      ngram = ctxt + [word]
      for i in range(N):
        train_ctxts[tuple(ctxt[i:])] += 1
        train_counts[tuple(ngram[i:])] += 1

alpha = [UNK_ALPHA, UNI_ALPHA, BI_ALPHA]
lls = 0
words = 0
with open(sys.argv[2], "r") as f:
  for line in f:
    sent = line.strip().split(" ") + ["<s>"]
    ngram = ["<s>"] * N
    for word in sent:
      ctxt = ngram[1:]
      ngram = ctxt + [word]
      total = alpha[0] / VOCAB_SIZE
      for i in range(N):
        if tuple(ngram[i:]) in train_counts:
          total += alpha[N-i] * train_counts[tuple(ngram[i:])] / train_ctxts[tuple(ctxt[i:])]
      lls += math.log(total)
    words += len(sent)-1  
my_score = math.exp(-lls/words)
print ("perplexity at alpha=%r: %f" % (alpha, my_score))

