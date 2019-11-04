import sys
from collections import defaultdict

col = int(sys.argv[1])

wid = defaultdict(lambda: len(wid))

x = wid["<eps>"]
for line in sys.stdin:
  arr = line.strip().split()
  if len(arr) == 5:
    x = wid[arr[col]]

it = list(wid.items())
for x, y in sorted(it, key=lambda x: x[1]):
  print(x, y)
