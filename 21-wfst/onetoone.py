import sys
import math
from collections import defaultdict

with open(sys.argv[1], 'r') as f:
  flines = [x.strip().split() for x in f.readlines()]

with open(sys.argv[2], 'r') as e:
  elines = [x.strip().split() for x in e.readlines()]

fecount = defaultdict(lambda: 0)
ecount = defaultdict(lambda: 0)
for fl, el in zip(flines, elines):
  for f, e in zip(fl, el):
    fecount[f,e] += 1
    ecount[e] += 1

for (f,e), val in fecount.items():
  print("0 0 %s %s %.4f" % (f, e, 0 if val == ecount[e] else -math.log(val/ecount[e])))
print("0 0 </s> </s> 0")
print("0")
