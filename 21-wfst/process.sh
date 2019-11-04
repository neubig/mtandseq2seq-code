#!/bin/bash
set -e

# Simple example FST
fstcompile --keep_isymbols --keep_osymbols --isymbols=example.isym --osymbols=example.isym example.txt example.fst
fstdraw example.fst example.dot
dot -Tps example.dot > example.ps
ps2pdf example.ps
pdfcrop example.pdf
mv example-crop.pdf example.pdf

# Create a bigram language model from the corpus
python bigram.py < corpuse.txt > bigram.txt
python symbols.py 2 < bigram.txt > bigram.isym

fstcompile --keep_isymbols --keep_osymbols --isymbols=bigram.isym --osymbols=bigram.isym bigram.txt bigram.fst
fstdraw --acceptor --show_weight_one --ssymbols=bigram.ssym bigram.fst bigram.dot
dot -Tps bigram.dot > bigram.ps
ps2pdf bigram.ps
pdfcrop bigram.pdf
mv bigram-crop.pdf bigram.pdf

# Create a one-to-one translation model
python onetoone.py corpusf.txt corpuse.txt > onetoone.txt
python symbols.py 2 < onetoone.txt > onetoone.isym
python symbols.py 3 < onetoone.txt > onetoone.osym

fstcompile --keep_isymbols --keep_osymbols --isymbols=onetoone.isym --osymbols=onetoone.osym onetoone.txt onetoone.fst
fstdraw --acceptor --show_weight_one onetoone.fst onetoone.dot
dot -Tps onetoone.dot > onetoone.ps
ps2pdf onetoone.ps
pdfcrop onetoone.pdf
mv onetoone-crop.pdf onetoone.pdf
open onetoone.pdf

# Compose together a translation model and languge model
fstcompile --keep_isymbols --keep_osymbols --isymbols=onetoone.isym --osymbols=bigram.isym onetoone.txt | fstarcsort --sort_type=olabel > onetoone.fst
fstcompose onetoone.fst bigram.fst composed.fst
fstdraw --show_weight_one composed.fst composed.dot
dot -Tps composed.dot > composed.ps
ps2pdf composed.ps
pdfcrop composed.pdf
mv composed-crop.pdf composed.pdf
open composed.pdf

# Formulate the input as a WFST
fstcompile --keep_isymbols --keep_osymbols --isymbols=onetoone.isym --osymbols=onetoone.isym input.txt input.fst
fstdraw --acceptor input.fst input.dot
dot -Tps input.dot > input.ps
ps2pdf input.ps
pdfcrop input.pdf
mv input-crop.pdf input.pdf
open input.pdf

# Compose together into a search graph
fstcompose input.fst composed.fst search.fst
fstdraw search.fst search.dot
dot -Tps search.dot > search.ps
ps2pdf search.ps
pdfcrop search.pdf
mv search-crop.pdf search.pdf
open search.pdf

# Remove epsilons to make it easier to read
fstrmepsilon search.fst searchrmeps.fst
fstdraw searchrmeps.fst searchrmeps.dot
dot -Tps searchrmeps.dot > searchrmeps.ps
ps2pdf searchrmeps.ps
pdfcrop searchrmeps.pdf
mv searchrmeps-crop.pdf searchrmeps.pdf
open searchrmeps.pdf

# Some extra examples of composing
python symbols.py 2 < t1.txt > t1.sym
python symbols.py 2 < t2.txt > t2.sym
python symbols.py 3 < t2.txt > t3.sym

fstcompile --keep_isymbols --keep_osymbols --isymbols=t1.sym --osymbols=t2.sym t1.txt t1.fst
fstdraw t1.fst t1.dot
dot -Tps t1.dot > t1.ps
ps2pdf t1.ps
pdfcrop t1.pdf
mv t1-crop.pdf t1.pdf
open t1.pdf

fstcompile --keep_isymbols --keep_osymbols --isymbols=t2.sym --osymbols=t3.sym t2.txt t2.fst
fstdraw t2.fst t2.dot
dot -Tps t2.dot > t2.ps
ps2pdf t2.ps
pdfcrop t2.pdf
mv t2-crop.pdf t2.pdf
open t2.pdf

fstcompose t1.fst t2.fst t3.fst
fstdraw --ssymbols=t3.ssym t3.fst t3.dot
dot -Tps t3.dot > t3.ps
ps2pdf t3.ps
pdfcrop t3.pdf
mv t3-crop.pdf t3.pdf
open t3.pdf

