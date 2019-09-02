import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('input', help='Path to the input file.')
parser.add_argument('output', help='Path to the output file.')
parser.add_argument('cutoff', help='Cutoff', type=int)
parser.add_argument('vocab', help='Path to write the vocabulary file')
args, unknown = parser.parse_known_args()

# Read in the input data
with open(args.input, "r") as f:
  lines = f.readlines()

word_counts = defaultdict(lambda: 0)
for line in lines:
	sent = line.strip().split(" ") + ["<s>"]
	for word in sent:
		word_counts[word] += 1

# Write the output
with open(args.output, "w") as f:
	for line in lines:
		sent = line.strip().split(" ")
		out = ' '.join([w  if word_counts[w]>args.cutoff else "<UNK>" for w in sent])
		f.write(f"{out}\n")

# Write the output vocab
with open(args.vocab, "w") as f:
	for word in word_counts:
		if word_counts[word] > args.cutoff:
			f.write(f"{word}\t{word_counts[word]}\n")



