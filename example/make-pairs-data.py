from __future__ import print_function
from argparse import ArgumentParser
import cPickle as pkl
import numpy as np
import json as js


def main(infile, outfile, err_file, embed_file, gate_file):
    print("Loading embeddings...")
    embed = pkl.load(open(embed_file, "rb"))
    print("Loading gates...")
    gates = pkl.load(open(gate_file, "rb"))
    data = []

    err_file = open(err_file, "w")
    out_file = open(outfile, "w")
    
    print("Beginning iteration...")
    for line in open(infile, "r"):
        try:
            word, antonym = line.strip().split(" ")
            if (word in embed) and (word in gates) and (antonym in embed):
                print(word, antonym, file = out_file)
            else:
                print("Invalid example!")
        except Exception as exc:
            print(type(exc), exc, file = err_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", action = "store", dest = "infile")
    parser.add_argument("-o", action = "store", dest = "outfile")
    parser.add_argument("-e", action = "store", dest = "embed_file")
    parser.add_argument("-g", action = "store", dest = "gate_file")
    parser.add_argument("-err", action = "store", dest = "err_file")
    args = parser.parse_args()
    main(args.infile, args.outfile, args.err_file,
         args.embed_file, args.gate_file)
