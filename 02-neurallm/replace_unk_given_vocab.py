import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('input', help='Path to the input file.')
parser.add_argument('output', help='Path to the output file.')
parser.add_argument('vocab', help='Path to write the vocabulary file')
args, unknown = parser.parse_known_args()

# Read in the vocabulary
vocab = defaultdict(lambda:0)
with open(args.vocab, "r") as f:
	for line in f:
		l = line.split('\t')
		vocab[l[0]] = int(l[1])


# Read in the input data
with open(args.input, "r") as f:
  lines = f.readlines()


# Write the output
with open(args.output, "w") as f:
	for line in lines:
		sent = line.strip().split(" ")
		out = ' '.join([w  if w in vocab else "<UNK>" for w in sent])
		f.write(out + '\n')

