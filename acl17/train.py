from __future__ import print_function
from argparse import ArgumentParser
from tqdm import tqdm
from pastalog import Log
from models import Vocabulary, Seq2Seq, GatedBilinearTransfer, IdentityTransfer

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import bz2
import json as js
import cPickle as pkl
import numpy as np
from math import ceil
from random import shuffle
from itertools import ifilter

def pad_batch_data(data, batch_size):
    batches = []
    num_batches = ceil(float(len(data))/float(batch_size))
    for i in range(int(num_batches)):
        data_batch = data[i*batch_size:(i+1)*batch_size]

        inputs, gates = [ex["input"] for ex in data_batch], [ex["context"] for ex in data_batch]
        input_lens = map(len, inputs)
        max_input = max(input_lens)
        inputs, gates = map(list, zip(*sorted(zip(inputs, gates), key=lambda pair: len(pair[0]), reverse=True)))
        input_batch = [Variable(th.LongTensor(ex + (max_input-len(ex))*[0]).unsqueeze(0))
                       for ex in inputs]
        gate_batch = [Variable(th.DoubleTensor(ex).unsqueeze(0)) for ex in gates]

        lookup = sorted(enumerate(input_lens), key=lambda pair: pair[1], reverse=True)
        lookup = sorted(enumerate([i for i, _ in lookup]), key=lambda pair: pair[1])
        lookup = Variable(th.LongTensor(list(zip(*lookup)[0])))
        old_lens = input_lens
        new_lens = sorted(old_lens, reverse=True)

        outputs = [ex["output"] for ex in data_batch]
        output_lens = [len(ex) - 1 for ex in outputs] 
        max_output = max(output_lens)
        output_batch = [Variable(th.LongTensor(ex[:-1] + (max_output + 1 - len(ex))*[0]).unsqueeze(0))
                        for ex in outputs]
        target_batch = [Variable(th.LongTensor(ex[1:] + (max_output + 1 - len(ex))*[0]).unsqueeze(0))
                        for ex in outputs]
        
        batch = ((th.cat(input_batch, 0), new_lens), th.cat(gate_batch, 0), lookup,
                 (th.cat(output_batch, 0), output_lens),
                 th.cat(target_batch, 0))
        batches.append(batch)
        
    return batches

def main(opts):
    if opts.gpu == -1:
        cuda = False
    else:
        cuda = th.cuda.is_available()
    
    print("Loading data.")
    data = js.load(open(opts.train_data, "r"))
    data = sorted(data, key=lambda x: len(x["output"]),
                  reverse=True)
    train_data = pad_batch_data(data, opts.batch_size)
    print("Creating transfer module.")
    if opts.transfer == "bilinear":
        transfer = GatedBilinearTransfer(in_dim=opts.hidden_dim, gate_dim=opts.embed_dim,
                                         hidden_dim=opts.gated_hidden_dim,
                                         out_dim=opts.hidden_dim,
                                         target=opts.transfer_target)
    elif opts.transfer == "feedforward":
        NotImplementedError("MLP not yet implemented.")
    else:
        transfer = IdentityTransfer()

    print("Loading vocabulary")
    js_vocab = js.load(open(opts.vocab_file, "r"))
    vocab_list = ["<PAD>"]
    for i in range(1, len(js_vocab["forward"])+1):
        vocab_list.append(js_vocab["forward"][str(i)])
        
    vocab = Vocabulary(vocab_list, opts.unk_idx)
    print("Loading weights")
    weights = th.from_numpy(np.load(opts.embed_file))
    weights = th.cat([th.zeros(1, opts.embed_dim).double(), weights])
    print("Creating model")
    model = Seq2Seq(in_vocab=vocab, out_vocab=vocab,
                    in_embed_dim=opts.embed_dim,
                    out_embed_dim=opts.embed_dim,
                    hidden_dim=opts.hidden_dim,
                    transfer=transfer)
    model.encoder.load_embeddings(weights, fix_weights=True)
    model.decoder.load_embeddings(weights, fix_weights=True)

    if opts.test_vocab:
        print("Testing vocab.")
        w2v_file = '/local/filespace/am2156/nncg/GoogleNews-vectors-negative300.bin'
        print("Loading word2vec.")
        w2v = word2vec.Word2Vec.load_word2vec_format(w2v_file, binary=True)
        print("Testing word2vec.")
        for word in tqdm(vocab_list, desc="Words"):
            if word in w2v:
                w2v_vec = w2v[word]
                word_idx = model.in_vocab.forward_dict[word]
                torch_vec = model.encoder.embedding.weight.data[word_idx].numpy()
                if not np.array_equal(w2v_vec, torch_vec):
                    print("Embedding test failure:", word)
            else:
                print("{} not in word2vec.".format(word))
                    
    criterion = nn.NLLLoss()
    if opts.objective=="dual":
        margin_criterion = nn.MarginRankingLoss()
        scorer_nll = nn.NLLLoss()
        def score_sequence(log_probs, seq_lens, targets):
            scores = []
            for i in range(len(seq_lens)):
                log_prob = log_probs[:,i,:]
                target = targets[i, :seq_lens[i]]
                score = scorer_nll(log_prob, target)
                scores.append(score)
            scores = th.cat(scores)
            return scores
        
    optimizer = optim.Adam(ifilter(lambda param: param.requires_grad,
                                   model.parameters()),
                           lr=1e-2)

    if cuda:
        model.cuda()
        criterion.cuda()
    
    log = Log('http://localhost:{}'.format(opts.log_port), opts.model_name)

    print("Beginning training.")
    for i in tqdm(range(opts.num_epochs)):
        shuffle(train_data)
        for j, ex in tqdm(enumerate(train_data, 1),
                          total=len(train_data)):
            (input, input_lens), gate, lookup, (output, output_lens), target = ex
            if cuda:
                input, gate, lookup = input.cuda(), gate.cuda(), lookup.cuda()
                output, target = output.cuda(), target.cuda()

            out = model(input, output, input_lens=input_lens, output_lens=output_lens, lookup=lookup, g=gate)
            flat_out = out.view(out.size(0)*out.size(1), out.size(2))
            flat_target = target.view(target.size(0)*target.size(1))
            loss = criterion(flat_out, flat_target)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            log.post('loss', value=loss.data[0], step=j)

    th.save(model, opts.model_name + ".model.pt")
   
if __name__=="__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--gpu", dest="gpu", default=-1)
    parser.add_argument("--train_data", dest="train_data",
                        default="data/train_data.json")
    parser.add_argument("--model_name", dest="model_name",
                        default="test")
    parser.add_argument("--num_epochs", dest="num_epochs", default=10, type=int)
    parser.add_argument("--batch_size", dest="batch_size", default=10, type=int)
    parser.add_argument("--embed_dim", dest="embed_dim", default=300, type=int)
    parser.add_argument("--embed_file", dest="embed_file", default="data/emb_vocab.npy")
    parser.add_argument("--hidden_dim", dest="hidden_dim", default=150, type=int)
    parser.add_argument("--gated_hidden_dim", dest="gated_hidden_dim",
                        default=150, type=int)
    parser.add_argument("--transfer", dest="transfer", default="identity")
    parser.add_argument("--transfer_target", dest="transfer_target",
                        default="h")
    parser.add_argument("--objective", dest="objective", default="single")
    parser.add_argument("--margin", dest="margin", default=0.7, type=float)
    parser.add_argument("--margin_weight", dest="margin_weight", default=1.5, type=float)
    parser.add_argument("--vocab_file", dest="vocab_file",
                        default="data/vocab.json")
    parser.add_argument("--test_vocab", dest="test_vocab", action="store_true", default=False)
    parser.add_argument("--unk_idx", dest="unk_idx", default=3, type=int)
    parser.add_argument("--log_port", dest="log_port", default=9000)
    
    opts = parser.parse_args()
    main(opts)
