from __future__ import print_function
from argparse import ArgumentParser
from tqdm import tqdm
from models import Vocabulary, Seq2Seq, GatedBilinearTransfer, IdentityTransfer

import torch as th
import torch.nn as nn
from torch.autograd import Variable

import bz2
import json as js
import cPickle as pkl
from operator import itemgetter
              

def main(opts):
    print("Loading model.")
    model = th.load(opts.model_name + ".model.t7")#pkl.load(bz2.BZ2File(opts.model_name + ".pkl.bz2", "rb"))
    #th.save(model, opts.model_name + ".model.t7")
    if opts.gpu == -1:
        model.cpu()
        model.float()
    nll = nn.NLLLoss()
    #meter = ClassErrorMeter(accuracy=True)
    print("Loading data.")
    eval_data = js.load(open(opts.eval_data, "r"))
    
    def score_sequence(score_model, input, output, gate):
        log_probs = score_model(input, output[:, :-1], g=gate)
        score = nll(log_probs.view(log_probs.size(0)*log_probs.size(1), log_probs.size(2)),
                    output[0, 1:])
        return score.data[0]

    print("Starting evaluation.")
    correct, incorrect = [], []
    for ex in tqdm(eval_data):
        if reduce(lambda a,b: a or b, map(lambda x: len(x) == 0, [ex["output"]] + [ex["input"]] + ex["noise"])):
            continue
        input = Variable(th.LongTensor(ex["input"])).unsqueeze(0)
        gate = Variable(th.FloatTensor(ex["context"])).unsqueeze(0)
        sequences = [Variable(th.LongTensor(seq)).unsqueeze(0) for seq in ex["noise"]]
        sequences.insert(0, Variable(th.LongTensor(ex["output"])).unsqueeze(0))
        
        if opts.gpu > -1:
            model.cuda()
            nll.cuda()
            gate = gate.cuda()
            input = input.cuda()
            for i in range(len(sequences)):
                sequences[i] = sequences[i].cuda()
                
        sequence_scores = []
        for sequence in sequences:
            sequence_scores.append(score_sequence(model, input, sequence, gate))


            
        ex_input = " ".join(model.in_vocab.decode(ex["input"]))
        ex_output = " ".join(model.out_vocab.decode(ex["output"])),
        model_output = model.generate(ex_input, method=opts.method,
                                      cuda = opts.gpu > -1, g=gate)

        max_idx, _ = max(enumerate(sequence_scores), key=itemgetter(1))
        if max_idx == 0:
            correct.append({"input": ex_input, "output": ex_output,
                            "model_output": model_output})
        else:
            chosen_seq = model.out_vocab.decode(ex["noise"][max_idx - 1])
            incorrect.append({"input": ex_input, "output": ex_output,
                              "chosen_sequence": chosen_seq,
                              "model_output": model_output})           
        
    print("Accuracy: {}".format(float(len(correct))/len(eval_data)))
    pkl.dump((correct, incorrect), bz2.BZ2File(opts.model_name + "_output.pkl.bz2", "wb"))
    
if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu", dest="gpu", default=-1, type=int)
    parser.add_argument("--eval_data", dest="eval_data",
                        default="data/dev_data.json")
    parser.add_argument("--batch_size", dest="batch_size", default=10, type=int)
    parser.add_argument("--model_name", dest="model_name",
                        default="test")
    parser.add_argument("--method", dest="method",
                        default="beam")
    opts = parser.parse_args()
    main(opts)
