import sys
import math
from collections import defaultdict

ctxts1 = 0.0
ctxts2 = defaultdict(lambda: 0.0)
count1 = defaultdict(lambda: 0.0)
count2 = defaultdict(lambda: 0.0)
for line in sys.stdin:
  vals = line.strip().split() + ["</s>"]
  ctxt = "<s>"
  for val in vals:
    ctxts1 += 1
    ctxts2[ctxt] += 1
    count1[val] += 1
    count2[(ctxt,val)] += 1
    ctxt = val

ALPHA=0.1

stateid = defaultdict(lambda: len(stateid))

# Print the fallbacks
print("%d %d <eps> <eps> %.4f" % (stateid["<s>"], stateid[""], -math.log(ALPHA)))
for ctxt, val in ctxts2.items():
  if ctxt != "<s>":
    print("%d %d <eps> <eps> %.4f" % (stateid[ctxt], stateid[""], -math.log(ALPHA)))

# Print the unigrams
for word, val in count1.items():
  v1 = val/ctxts1
  print("%d %d %s %s %.4f" % (stateid[""], stateid[word], word, word, -math.log(v1)))

# Print the unigrams
for (ctxt, word), val in count2.items():
  v1 = count1[word]/ctxts1
  v2 = val/ctxts2[ctxt]
  val = 0.9 * v2 + 0.1 * v1
  print("%d %d %s %s %.4f" % (stateid[ctxt], stateid[word], word, word, -math.log(val)))

# Print the final state
print(stateid["</s>"]) 
  
