from __future__ import print_function
import cPickle as pkl
import json as js
import numpy as np
import os.path as op
from collections import Counter
from argparse import ArgumentParser
from gensim.models import word2vec
from string import punctuation
from tqdm import tqdm

def not_punct(s): return not (s in punctuation) 

def in_vocab(example, voc):
    return  reduce(lambda x, y: x and y, map(lambda c: c in voc, filter(not_punct, example["FULL"].split(" "))))

# def filter_vocab(vocab_count, n, model):
#     return [key for key in sorted(vocab_count, reverse = True, key = vocab_count.get)
#                 if (((key in model) or (key in ["a", "and", "of", "to"])) and (vocab_count[key]>n))]

# def filter_data(voc, nps):
#     exs = [key for key in nps if not_punct(nps[key])]
#     return exs

def main(np_file, vocab_file, index_file):
    print("Loading word2vec...")
    model = word2vec.Word2Vec.load_word2vec_format('/local/filespace/am2156/nncg/GoogleNews-vectors-negative300.bin', binary=True)
    wiki = pkl.load(open(np_file, "rb"))
    model_dim = 300

    print("Accumulating word frequencies...")
    vocab_count = Counter()
    for key in wiki:
        vocab_count.update(wiki[key]["FULL"].split(" "))
        vocab_count.update(wiki[key]["TITLE"].split("\xc2\xa0"))
        
    print("Constructing vocab dictionaries...")
    vocab = {"forward": {1: "<START>", 2: "<END>", 3: "<UNK>"},
             "reverse": {"<START>": 1, "<END>": 2, "<UNK>": 3}}
    embedding = []
    embedding.append(np.random.randn(1, model_dim))
    embedding.append(np.random.randn(1, model_dim))
    embedding.append(np.random.randn(1, model_dim))

    stop_words = ["a", "and", "of", "to"]
    voc = [word for word in vocab_count if not_punct(word) and ((word in model) or (word in stop_words))]
    for num, word in enumerate(voc, 4):
        vocab["forward"][num] = word
        vocab["reverse"][word] = num
        print("word:", word, "num:", num)
        if word in stop_words:
            embedding.append(np.expand_dims(model[word.capitalize()], axis = 0))
        else:
            embedding.append(np.expand_dims(model[word], axis = 0))

    embedding = np.concatenate(tuple(embedding), axis = 0)

    print("Saving vocab...")
    pkl.dump(vocab_count,
             open(op.join(op.dirname(vocab_file),
                          "counts_"+op.basename(vocab_file).split(".json")[0]+".pkl"),
                  "w"))
    js.dump(vocab, open(vocab_file, "w"))
    np.save(op.join(op.dirname(vocab_file),
                         "emb_"+op.basename(vocab_file).split(".json")[0]),
            embedding)

    print("Accumulating data...")
    data = []
    # data_keys = filter_data(set(voc), wiki)
    # print("Length vocab:", len(voc), "Length data: ", len(data_keys))
    lookup_idx = lambda k: vocab["reverse"].get(k, 3)
    for key in tqdm(wiki):
        rest = map(lookup_idx, filter(not_punct, wiki[key]["CONTEXT"].split(" ")))
        rest = reduce(lambda a, b: a + b, map(lambda k: embedding[k-1], rest)).tolist()
        example = {"input": map(lookup_idx, filter(not_punct, key[0].split(" "))),
                   "output": [1] + map(lookup_idx, filter(not_punct, key[1].split(" "))) + [2],
                   "context": rest}
        data.append(example)

    print("Saving data...")
    js.dump(data, open(index_file, "w"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nps", action = "store", default = "/local/filespace/am2156/nncg/wikipedia-nps.pkl",
                        dest = "np_file")
    parser.add_argument("--vocab", action = "store", dest = "vocab_file")
    parser.add_argument("--indices", action = "store", dest = "index_file")
    args = parser.parse_args()
    main(args.np_file, args.vocab_file, args.index_file)


