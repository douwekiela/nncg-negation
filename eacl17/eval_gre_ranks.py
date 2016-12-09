#!/usr/bin/env python

import sys
import numpy as np
from collections import defaultdict
from gensim import corpora, models, similarities
from gensim.models import word2vec
import re
from scipy.spatial.distance import cosine

antonym_goldfile = sys.argv[1] # file with gold antonym pairs, e.g. /local/filespace/lr346/disco/experiments/negation/nncg-negation/mohammad/gre_devset.txt
eval_file = sys.argv[2]  # output file to be evaluated (formatted torch output)

print("Loading model...")
model = word2vec.Word2Vec.load_word2vec_format('/home/ltf24/trunk/GoogleNews-vectors-negative300.bin', binary=True)
print("...Done")

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
choices = defaultdict(list)
with open(antonym_goldfile) as infile:
    for line in infile:
        m = re.match('(\S+): (\S+ \S+ \S+ \S+ \S+) :: (\S+)', line)
        antonyms[m.group(1)] = m.group(3)
        choices[m.group(1)] = m.group(2).split()

pred = {}
with open(eval_file) as infile:
    for line in infile:
        line = line.split("\t")
        foo = line[1].rstrip()
        pred[line[0]] = np.array([float(x) for x in foo.lstrip('[').rstrip(', ]').split(',')])

correct = 0
total = 0
for w in antonyms:
    print("INPUT:", w, "TARGET:", antonyms[w])
    if w not in model or w not in pred:
        continue
    cosines = {}
    for x in choices[w]:
        if x not in model:
            continue
        cosines[x] = 1 - cosine(pred[w], model[x])
    if antonyms[w] not in cosines:
        continue
    if antonyms[w] == max(cosines, key=cosines.get):
        correct += 1
    total += 1
    for x in cosines:
        print "   ", x, cosines[x],
        if x == antonyms[w] and x == max(cosines, key=cosines.get):
            print " ****"
        elif x == max(cosines, key=cosines.get):
            print " xxxx"
        else:
            print

print "Total correct", correct, "out of", total
