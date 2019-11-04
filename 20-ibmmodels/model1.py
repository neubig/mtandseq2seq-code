import sys
import math
import numpy as np
from collections import defaultdict

NUM_ITERS = 20;
TERMINATE = 1/1000;
CUTOFF = 1e-5;

if len(sys.argv) != 3:
  print("Usage: model1.pl FFILE EFILE\n")

def loadcorp(fname, add_null=False):
  wmap = defaultdict(lambda: len(wmap))
  if add_null:
    nid = wmap["NULL"]
  corp = []
  with open(fname, 'r') as f:
    for line in f:
      orig = [nid] if add_null else []
      corp.append(orig + [wmap[x] for x in line.strip().split()])
  warr = list(range(len(wmap)))
  for k, v in wmap.items():
    warr[v] = k
  return corp, warr

fcorp, fsyms = loadcorp(sys.argv[1])
ecorp, esyms = loadcorp(sys.argv[2], add_null=True)
assert(len(fcorp) == len(ecorp))
fcount = sum([len(fsent) for fsent in fcorp])

print(f"Loaded {len(fcorp)} sentences", file=sys.stderr);

# initialize to uniform
uniprob = 1/float(len(fsyms))
t = {}
for fsent, esent in zip(fcorp, ecorp):
  for f in fsent:
    for e in esent:
      t[f,e] = uniprob

# train t
lastll = None
for myiter in range(NUM_ITERS):
  count = defaultdict(lambda: 0)
  total = np.zeros(len(esyms))
  ll = 0
  # E step
  for fsent, esent in zip(fcorp, ecorp):
    stotal = defaultdict(lambda: 0)
    tfe = np.zeros( (len(fsent), len(esent)) )
    for fi, f in enumerate(fsent):
      for ei, e in enumerate(esent):
        tfe[fi,ei] = t[f,e]
        stotal[f] += tfe[fi,ei]
      ll += math.log(stotal[f]/len(esent))
    for fi, f in enumerate(fsent):
      for ei, e in enumerate(esent):
        count[f,e] += tfe[fi,ei]/stotal[f]
        total[e] += tfe[fi,ei]/stotal[f]
  # M step
  for (f,e), v in count.items():
    t[f,e] = count[f,e]/total[e]
  print(f'Iter {myiter}: ll={ll/fcount}', file=sys.stderr)
  if lastll and (lastll-ll) > ll * TERMINATE:
    break
  lastll = ll

for f, e in sorted(t.keys()):
  tfe = t[f,e]
  if tfe > CUTOFF:
    print(f'{fsyms[f]} {esyms[e]} {tfe}')
