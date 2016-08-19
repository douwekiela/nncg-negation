#!/usr/bin/env python

import sys
import cPickle as pickle
import numpy as np

gate_dictfile = sys.argv[1]    # file with dictionary containing vectors to make the gate, e.g. /local/filespace/ltf24/negation/a.words.top10nns.wordsonly.pkl
antonym_goldfile = sys.argv[2] # file with gold antonym pairs, e.g. /local/filespace/ltf24/negation/a.pairs.embed
embed_file = sys.argv[3]       # file with word embeddings, e.g. /local/filespace/ltf24/negation/a.words.embed.pkl
outfile = sys.argv[4]          # output file for training data in torch format

antonyms = {}
with open(antonym_goldfile) as infile:
    for line in infile:
        line = line.split()
        antonyms[line[0]] = line[1]

# 
# Each entry in the table is {input word vector, gate vector, target antonym word vector}
# 
embeddings = pickle.load(open(embed_file, 'rb'))
gate_dict = pickle.load(open(gate_dictfile, 'rb'))
with open(outfile, 'w') as outf:
    outf.write("traindata = {}\n")
    cntr = 0
    for w in antonyms:
        cntr += 1
        vecs = []
        for neigh in gate_dict[w]:
            vecs.append(gate_dict[w][neigh])
        outf.write("traindata[" +  str(cntr) + "] = {torch.Tensor({")
        for x in embeddings[w]:                    # input word vector
            outf.write(str(x) + ",")
        outf.write("}), torch.Tensor({")
        for x in np.sum(np.array(vecs), axis=0):   # gate = sum of related words in the class
            outf.write(str(x) + ",")
        outf.write("}), torch.Tensor({")
        for x in embeddings[antonyms[w]]:          # target antonym word vector
            outf.write(str(x) + ",")
        outf.write("})}\n")

