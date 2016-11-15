import cPickle as pickle
import json as js
import numpy as np
import sys


nns = pickle.load(open(sys.argv[1], 'rb')) # dictionary of related words of choice

lookup = {}
for key in nns:
    vecs = []
    for neigh in nns[key]:
        vecs.append(nns[key][neigh])
    lookup[key] = np.sum(np.array(vecs), axis=0).tolist() # gate = sum of related words in the class

js.dump(lookup, open(sys.argv[2],'w')) # dictionary of gate vectors for each adjective


