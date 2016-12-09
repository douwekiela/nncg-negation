from gensim import corpora, models, similarities
from gensim.models import word2vec
from argparse import ArgumentParser
import numpy as np
import cPickle as pickle
import json as js
import sys
from collections import defaultdict


def topNeighbours(word, model):
	top =  model.most_similar(positive=[word], topn=100)
	ctr = 0
	nns = []		
	ind = 0
	while ctr < 10 and ind < 100:
		neigh = top[ind][0].encode('ascii','ignore')
		# not MWE and word in mikolov pretrained		
		if ('_' not in neigh) and (neigh in model): 
			nns.append(neigh)
			ctr +=1		
		ind += 1
	
	return nns


def main(infile, outfile, nnfile, get_nn):
        lkp = {}
        nn_lkp = defaultdict(dict)

        f = open(infile, "r")

        print "Loading model..."
        model = word2vec.Word2Vec.load_word2vec_format('/home/ltf24/trunk/GoogleNews-vectors-negative300.bin', binary=True)
        print "Loaded model..."


        ctr = 0
        for line in f.readlines():
                print "Reading word %s" %(ctr)
                words = line.strip().split() 
                input_word = words[0]

                if (input_word in model):

                        vec = np.array(np.nan_to_num(model[input_word]))
                        lkp[input_word] = vec.tolist()
                        if get_nn:
                                nns = topNeighbours(input_word, model)
                                for i in range(len(nns)):
                                        nn_lkp[input_word][nns[i]] = model[nns[i]]
                ctr+=1


        js.dump(lkp, open(outfile,'w'))
        if get_nn:
                pickle.dump(nn_lkp, open(nnfile, 'wb'))

if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("-i", action = "store", dest = "infile")
        parser.add_argument("-o", action = "store", dest = "outfile")
        parser.add_argument("-nn", action="store_true",
                            dest = "get_nn", default = False)
        parser.add_argument("-nf", action = "store", dest = "nnfile")
        args = parser.parse_args()
        main(args.infile, args.outfile, args.nnfile, args.get_nn)


