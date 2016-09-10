#!/usr/bin/env python

import sys
print("1")
import numpy as np
print("2")
from collections import defaultdict
print("3")
from gensim import corpora, models, similarities
print("4")
from gensim.models import word2vec
print("5")

antonym_goldfile = sys.argv[1] # file with gold antonym pairs, e.g. /local/filespace/ltf24/negation/a.pairs.embed
eval_file = sys.argv[2]  # output file to be evaluated (formatted torch output)

print("Loading model...")
model = word2vec.Word2Vec.load_word2vec_format('/home/ltf24/trunk/GoogleNews-vectors-negative300.bin', binary=True)
print("Loaded model...")

def topNeighbours(vec):
    top =  model.similar_by_vector(vec, topn=100)
    ctr = 0
    nns = []
    ind = 0
    while ctr < 10 and ind < 100:
        neigh = top[ind][0].encode('ascii','ignore')
        # not MWE and word in mikolov pretrained
        if ('_' not in neigh) and (neigh in model): 
            nns.append(neigh)
            ctr += 1
        ind += 1
    return nns

antonyms = {}
with open(antonym_goldfile) as infile:
    for line in infile:
        line = line.split()
        antonyms[line[0]] = line[1]

pred = {}
with open(eval_file) as infile:
    for line in infile:
        line = line.split("\t")
        foo = line[1].rstrip()
        pred[line[0]] = np.array([float(x) for x in foo.lstrip('[').rstrip(', ]').split(',')])

for w in pred:
    print("INPUT:", w, "TARGET:", antonyms[w])
    print("   "),
    print(topNeighbours(pred[w]))

