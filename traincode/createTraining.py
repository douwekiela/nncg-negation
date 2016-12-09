#!/usr/bin/env python

import pickle
import argparse
from collections import defaultdict
import numpy as np
from nltk.corpus import wordnet as wn
from gensim.models import word2vec
from gensim import corpora, models, similarities
import random
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--outfilebase', help='Base file name for output; will be appended with .train and .test')

ap.add_argument('--testwords', help='File with list of test input words to exclude from training data')
ap.add_argument('--filter_gate_words', help='If true, not only the test input words but also their gate words are filtered out of the training data', action="store_true")
ap.add_argument('--use_wn_gates', help='If true, use WN gates and back off to nearest neighbours; if false, use only nearest neighbours in gates', action="store_true")

ap.add_argument('--mingate', help = 'Minimum threshold for number of terms in WordNet gates', default = 10)
ap.add_argument('--logging', help='Whether to log which words are used', action="store_true")
ap.add_argument('--gensim', help = 'Gensim model, e.g. /home/ltf24/trunk/GoogleNews-vectors-negative300.bin', default='/home/ltf24/trunk/GoogleNews-vectors-negative300.bin')
ap.add_argument('--wantonyms', help = 'File with WordNet antonym pairs, e.g. /local/filespace/ltf24/negation/a.pairs.embed', default='/local/filespace/lr346/disco/experiments/negation/nncg-negation/example/a.pairs.embed')
args = ap.parse_args()

args.mingate = int(args.mingate)

SEED = 4623
rg = random.Random()

print("Creating training data")
print(args)

if(args.logging):
    outfile_logtrain = open(args.outfilebase+'.logtrain', 'w')
    outfile_logtest = open(args.outfilebase+'.logtest', 'w')

print("Loading full gensim w2v model")
model = word2vec.Word2Vec.load_word2vec_format(args.gensim, binary=True)
print("...Done")

print("Loading WordNet antonyms...")
wnAntonyms = defaultdict(str)
with open(args.wantonyms) as infile:
    for line in infile:
        line = line.split()
        wnAntonyms[line[0]] = line[1]
        wnAntonyms[line[1]] = line[0]    # most reversed cases are in, but a few are missing
print("...Done")

print("Loading test words for exclusion...")
test_words = []
with open(args.testwords) as infile:
    for line in infile:
        line = line.rstrip()
        test_words.append(line)
print("...Done")

def cohyponyms(w):
    names = defaultdict(int)

    if wn.synset(w+'.a.1'):
        adjtype = 'a'
    elif wn.synset(w+'.s.1'):
        adjtype = 's'
    else:
        return names.keys()

    if wn.synset(w+'.'+adjtype+'.1'):
        # get other lemmas from synset
        for lemma in wn.synset(w+'.'+adjtype+'.1').lemmas():
            if lemma.name() != w and lemma.name() in model and len(lemma.name()) > 0: # dont include input word in list of related words    
                names[lemma.name()] += 1

        # check for "parent (attribute)"
        if wn.synset(w+'.'+adjtype+'.1').attributes():
            rel_synsets = (wn.synset(w+'.'+adjtype+'.1').attributes())[0].attributes()
            for synset in rel_synsets:
                for lemma in synset.lemmas():
                    if lemma.name() != w and lemma.name() in model and len(lemma.name()) > 0: # dont include input word in list of related words    
                        names[lemma.name()] += 1
                        
        # check for similar words                                           
        if wn.synset(w+'.'+adjtype+'.1').similar_tos():
            rel_synsets = wn.synset(w+'.'+adjtype+'.1').similar_tos()
            for synset in rel_synsets:
                for lemma in synset.lemmas():
                    if lemma.name() != w and lemma.name() in model and len(lemma.name()) > 0: # dont include input word in list of related words 
                        names[lemma.name()] += 1
                    for ant in lemma.antonyms():
                        if ant.name() != w and ant.name() in model and len(ant.name()) > 0:
                            names[ant.name()] += 1

        # check for antonyms
        if (wn.synset(w+'.'+adjtype+'.1').lemmas())[0].antonyms():                 
            rel_lemmas = (wn.synset(w+'.'+adjtype+'.1').lemmas())[0].antonyms()
            for lemma in rel_lemmas:                                    
                if lemma.name() != w and lemma.name() in model and  len(lemma.name()) > 0: # dont include input word in list of related words    
                    names[lemma.name()] += 1
                    for x in synonyms(lemma.name()):
                        names[x] += 1

    return names.keys()

def synonyms(w):
    names = defaultdict(int)

    if wn.synset(w+'.a.1'):
        adjtype = 'a'
    elif wn.synset(w+'.s.1'):
        adjtype = 's'
    else:
        return names.keys()

    # get other lemmas from synset
    for lemma in wn.synset(w+'.'+adjtype+'.1').lemmas():
        if lemma.name() != w and lemma.name() in model and len(lemma.name()) > 0: # dont include input word in list of related words    
            names[lemma.name()] += 1

    # check for similar words                                           
    if wn.synset(w+'.'+adjtype+'.1').similar_tos():    # most frequent sense of word as adjective
        rel_synsets = wn.synset(w+'.'+adjtype+'.1').similar_tos()
        for synset in rel_synsets:
            for lemma in synset.lemmas():
                if lemma.name() != w and lemma.name() in model and len(lemma.name()) > 0: # dont include input word in list of related words 
                    names[lemma.name()] += 1
    return names.keys()

def antonyms(w):
    names = defaultdict(int)

    if wn.synset(w+'.a.1'):
        adjtype = 'a'
    elif wn.synset(w+'.s.1'):
        adjtype = 's'
    else:
        return names.keys()

    # check for antonyms
    if (wn.synset(w+'.'+adjtype+'.1').lemmas())[0].antonyms():                 
        rel_lemmas = (wn.synset(w+'.'+adjtype+'.1').lemmas())[0].antonyms()
        for lemma in rel_lemmas:                                    
            if lemma.name() != w and lemma.name() in model and  len(lemma.name()) > 0: # dont include input word in list of related words    
                names[lemma.name()] += 1
                for x in synonyms(lemma.name()):
                    names[x] += 1
    return names.keys()

def topNeighbours(w):
    if w not in model:
        return []
    top = model.most_similar(positive=[w], topn=100)
    nns = []
    for i in range(len(top)):
        neigh = top[i][0].encode('ascii', 'ignore')
        if '_' not in neigh and  len(neigh) > 0 and neigh in model:    # gensim can return similar words that are not in the model
            nns.append(neigh)
            if len(nns) == 10:
                break
    return nns

def get_gate_words(w):
    gw = []
    if args.use_wn_gates:
        for x in cohyponyms(w):
            gw.append(x)
    if len(gw) < args.mingate:
        for x in topNeighbours(w):
            if x not in gw:
                gw.append(x)
            if len(gw) >= args.mingate:
                break
    return gw


print "Getting gates"
gateWords = defaultdict(lambda: defaultdict(int))   # word => gate word => count
gateVecs = defaultdict(np.array)  # word => gate vector

# at this point wnAntonyms.keys() contains all of the basic WN adjectives we're using.
# we'll add the synonyms of the antonyms to the training data as targets, though, using antonyms() from this file.

all_antonyms = defaultdict(lambda: defaultdict(int))   # word => antonym => cnt
for w in wnAntonyms:   # note the keys contain all the basic WN adjectives
    for y in antonyms(w):
        all_antonyms[w][y] += 1
        all_antonyms[y][w] += 1

for w in all_antonyms:  # note the keys contain all the adjectives involved in the extended set of antonyms
    gateWords[w] = get_gate_words(w) 
for w in test_words:
    if w not in gateWords:
        gateWords[w] = get_gate_words(w)
        
for w in gateWords:
    if len(gateWords[w]) > 0:
        gateVecs[w] = np.sum(np.array([model[x] for x in gateWords[w]]), axis=0)/len(gateWords[w])
print "...Done"

extended_test_words = []
if args.filter_gate_words:
    ext = defaultdict(int)
    print("Filtering out gate words of all test words")
    for w in test_words:
        ext[w] += 1
        for x in gateWords[w]:
            ext[x] += 1
    extended_test_words = ext.keys()
    print("...Done")
else:
    print("Not filtering out gate words")
    extended_test_words = test_words

print "Filtering training pairs"
training_pairs = defaultdict(lambda: defaultdict(int))
for w in all_antonyms:
    if w in model:
        for y in antonyms(w):
            if y in model:
                if w not in extended_test_words and w in gateVecs:
                    training_pairs[w][y] += 1
                if y not in extended_test_words and y in gateVecs:
                    training_pairs[y][w] += 1
print "...Done"

print "Printing training data"
with open(args.outfilebase + '.train', 'w') as outtrain:
    for w in training_pairs:
        for y in training_pairs[w]:
            outtrain.write(w + " ")               # input word
            for x in model[w]:                    # input word vector
                outtrain.write(str(x) + " ")
            for x in gateVecs[w]:                 # gate vector of TRAIN WORD
                outtrain.write(str(x) + " ")
            for x in model[y]:                    # target antonym word vector
                outtrain.write(str(x) + " ")
            outtrain.write("\n")
            if args.logging:
                outfile_logtrain.write(w + " " + y + "\n")
                
print "...Done"

print "Printing test data"
with open(args.outfilebase + '.test', 'w') as outtest:
    for w in test_words:
        if w not in model:
            continue
        outtest.write(w + " ")                # input test word
        for x in model[w]:                    # input word vector
            outtest.write(str(x) + " ")
        for x in gateVecs[w]:                 # gate vector of input test word
            outtest.write(str(x) + " ")
        outtest.write("\n")
        if args.logging:
            outfile_logtest.write(w + "\n")

print("...Done")
