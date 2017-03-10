from __future__ import print_function

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
#from torchtext import vocab

class Vocabulary(object):
    def __init__(self, tokens, unk_idx):
        self.tokens = tokens
        self.unk_idx = unk_idx
        self.vocab_size = len(tokens)
        self.forward_dict = dict((token, i) for i, token in enumerate(tokens))
        self.backward_dict = dict(enumerate(tokens))

    def encode(self, tokens):
        return [self.forward_dict.get(token, self.unk_idx) for token in tokens]

    def decode(self, ids):
        return [self.backward_dict.get(idx, "<UNK>") for idx in ids]

    def batch_encode(self, inputs):
        batch = [self.encode(token) for token in inputs]
        max_len = max(map(len, batch))
        batch = [ids + (max_len - len(ids))*[0] for ids in batch]
        return batch
    
    def __len__(self):
        return len(self.tokens)
    
    

class Bilinear(nn.Module):
    """
    Documentation for Bilinear
    """
    def __init__(self, first_dim, second_dim, out_dim):
        super(Bilinear, self).__init__()
        self.first_dim = first_dim
        self.second_dim = second_dim
        self.out_dim = out_dim
        self.weights = nn.Parameter(data=th.randn(first_dim, second_dim, out_dim).double(),
                                    requires_grad=True)

    def forward(self, input1, input2):
        # preconditions
        assert input1.ndimension() == 2, "Inputs must be matrices (2-dimensional). Input 1 has {} dimensions.".format(input1.ndimension())
        assert input2.ndimension() == 2, "Inputs must be matrices (2-dimensional). Input 2 has {} dimensions.".format(input2.ndimension())
        assert input1.size(1) == self.first_dim, "Input 1's shape is inconsistent with the bilinear weight matrix."
        assert input2.size(1) == self.second_dim, "Input 2's shape is inconsistent with the bilinear weight matrix."
        assert input1.size(0) == input2.size(0), """Input batch sizes must be equal. 
               Input 1 has batch size {}, while input 2 has batch size {}.""".format(input1.size(0), input2.size(0))

        # computation
        batch_size = input1.size(0)
        input1_expanded = input1.unsqueeze(2).unsqueeze(3).expand(batch_size, self.first_dim,
                                                                  self.second_dim, self.out_dim)
        input2_expanded = input2.unsqueeze(1).unsqueeze(3).expand(batch_size, self.first_dim,
                                                                  self.second_dim, self.out_dim)
        weights_expanded = self.weights.unsqueeze(0).expand(batch_size, self.first_dim,
                                                            self.second_dim, self.out_dim)

        output = (weights_expanded*input1_expanded*input2_expanded).sum(1).sum(2)
        return output.squeeze(1).squeeze(1)
        

class EncoderRNN(nn.Module):
    """
    Documentation for EncoderRNN
    """
    def __init__(self, vocab, embed_dim, hidden_dim):
        super(EncoderRNN, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.embedding.double()
        self.rnn = nn.LSTM(embed_dim, hidden_dim,
                           batch_first=True, bias=False)
        self.rnn.double()
        
    def forward(self, input, h0, c0, lens=None):
        embedded = self.embedding(input)
        if lens:
            embedded = pack_padded_sequence(embedded, lens, batch_first=True)
        output, (hn, cn) = self.rnn(embedded, (h0, c0))
        return output, hn, cn

    def load_embeddings(self, weights, fix_weights=True):
        self.embedding.weight.data = weights
        if fix_weights:
            self.embedding.weight.requires_grad = False


class DecoderRNN(nn.Module):
    """
    Documentation for DecoderRNN
    """
    def __init__(self, vocab, start_idx, end_idx, embed_dim, hidden_dim):
        super(DecoderRNN, self).__init__()
        self.vocab = vocab
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.encoder = EncoderRNN(vocab, embed_dim, hidden_dim)
        self.scorer = nn.Sequential(nn.Linear(hidden_dim, len(vocab)), 
                                    nn.LogSoftmax())
        self.scorer.double()

    def load_embeddings(self, weights, fix_weights=True):
        self.encoder.load_embeddings(weights, fix_weights)
        
    def forward(self, input, h0, c0, lens=None):
        output, hn, cn = self.encoder(input, h0, c0, lens)
        if lens:
            output, _ = pad_packed_sequence(output)
        logprobs = self.scorer(output.contiguous().view(output.size(0)*output.size(1), output.size(2)))
        logprobs = logprobs.view(output.size(0), output.size(1), logprobs.size(1))
        return logprobs, hn, cn
    

    def generate(self, h0, c0, method="beam", **kwargs):
        generator = {"greedy": self.greedy_decode,
                     "beam": self.beam_decode,
                     "sample": self.temperature_sample}.get(method)
        ids = generator(h0, c0, **kwargs)
        tokens = self.vocab.decode(ids)
        return tokens
    
    def temperature_sample(self, h0, c0, temp=1, max_length=20, **kwargs):
        pass

    def greedy_decode(self, h0, c0, max_length=20, **kwargs):
        pass

    def beam_decode(self, h0, c0, beam_size=5, max_length=10, cuda=False, **kwargs):
        def get_ij(idx, n):
            j = idx % n
            i = (idx - j)/n
            return i, j
        beam = []
        completed = []
        prune_factor = float("-inf")

        start_symbol = Variable(th.LongTensor([self.start_idx]))
        beam_symbols = start_symbol.unsqueeze(1)
        if cuda:
            start_symbol = start_symbol.cuda()
            beam_symbols = beam_symbols.cuda()
        scores, out_h, out_c = self.forward(beam_symbols, h0, c0)
        top_scores, top_ids = scores.view(scores.numel()).sort(0, True)
        _, dim_beam, dim_vocab = scores.size()

     
        for idx in range(min(beam_size, dim_vocab)):
            i, j = get_ij(top_ids[idx], dim_vocab)
            if cuda:
                j = j.cuda()
            seq = th.cat([start_symbol, j])
            score = top_scores[idx]
            if j.data[0] == self.end_idx:
                completed.append({"seq": seq.data.tolist(), "score": score})
                prune_factor = top_scores[idx].data[0]
            else:
                beam.append({"seq": seq, "h": out_h[:, 0, :],
                             "c": out_c[:, 0, :], "score": score})

        count = 0
        while len(beam) > 0 and  count < max_length:
            beam_symbols = th.cat([item["seq"][-1].unsqueeze(1) for item in beam], 0)
            beam_h = th.cat([item["h"].unsqueeze(1) for item in beam], 1)
            beam_c = th.cat([item["c"].unsqueeze(1) for item in beam], 1)

      
            log_probs, out_h, out_c = self.forward(beam_symbols, beam_h, beam_c)
            dim_beam, _, dim_vocab = log_probs.size()
            beam_scores = th.cat([item["score"] for item in beam]).unsqueeze(1).unsqueeze(1)
            beam_scores = beam_scores.expand(dim_beam, 1, dim_vocab)
            scores = beam_scores + log_probs
            top_scores, top_ids = scores.view(scores.numel()).sort(0, True)

       

            new_beam = []
            for idx in range(min(beam_size, len(beam))):
                score = top_scores[idx]
                i, j = get_ij(top_ids[idx], dim_vocab)

                if (score.data[0] >= prune_factor):
                    seq = th.cat([beam[i.data[0]]["seq"], j])
                    if j.data[0] == self.end_idx:
                        completed.append({"seq": seq.data.tolist(), "score": score})
                        prune_factor = score.data[0]
                    else:
                        new_beam.append({"seq": seq, "h": out_h[:, i.data[0], :],
                                         "c": out_c[:, i.data[0], :], "score": score})
                else:
                    break

            beam = new_beam
            count += 1
            
        return completed[-1]["seq"]

class Seq2Seq(nn.Module):
    """
    Documentation for Seq2Seq
    """
    def __init__(self, in_vocab, out_vocab, in_embed_dim,
                 out_embed_dim, hidden_dim, transfer):
        super(Seq2Seq, self).__init__()
        self.in_vocab = in_vocab
        self.out_vocab = out_vocab
        self.hidden_dim = hidden_dim
        self.h0 = nn.Parameter(th.randn(1, 1, hidden_dim).double())
        self.c0 = nn.Parameter(th.randn(1, 1, hidden_dim).double())
        self.encoder = EncoderRNN(in_vocab, in_embed_dim, hidden_dim)
        self.decoder = DecoderRNN(out_vocab, 1, 2,
                                  out_embed_dim, hidden_dim)
        self.transfer = transfer

    def forward(self, input, output, input_lens=None, output_lens=None, lookup=None, **kwargs):
        h0 = self.h0.expand(1, input.size(0), self.hidden_dim).contiguous()
        c0 = self.c0.expand(1, input.size(0), self.hidden_dim).contiguous()
        input_encoded, input_h, input_c = self.encoder(input, h0, c0, lens=input_lens)

        if lookup:
            input_h = th.index_select(input_h, 1, lookup)
            input_c = th.index_select(input_c, 1, lookup)
            
        transfer_h, transfer_c = self.transfer(input_h, input_c, **kwargs)
        log_probs, _, _ = self.decoder(output, transfer_h, transfer_c, lens=output_lens)
        return log_probs

    def generate(self, input_seq, method="beam", cuda=False, **kwargs):
        input_ids = self.in_vocab.encode(input_seq.split(" "))
        input = Variable(th.LongTensor(input_ids)).unsqueeze(0)
        h0 = Variable(th.zeros(1, 1, self.hidden_dim).contiguous())
        c0 = Variable(th.zeros(1, 1, self.hidden_dim).contiguous())
        if cuda:
            input = input.cuda()
            h0 = h0.cuda()
            c0 = c0.cuda()
        input_encoded, input_h, input_c = self.encoder(input, h0, c0)
        transfer_h, transfer_c = self.transfer(input_h, input_c, **kwargs)
        output = self.decoder.generate(transfer_h, transfer_c, method=method, cuda=cuda, **kwargs)
        return " ".join(output)


class IdentityTransfer(nn.Module):
    def __init__(self):
        super(IdentityTransfer, self).__init__()

    def forward(self, h, c, **kwargs):
        return h, c

class GatedBilinearTransfer(nn.Module):
    def __init__(self, in_dim, gate_dim, hidden_dim,
                 out_dim, target="h"):
        super(GatedBilinearTransfer, self).__init__()
        self.target = target
        self.in_bilinear = Bilinear(in_dim, gate_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.out_bilinear = Bilinear(hidden_dim, gate_dim, out_dim)

    def forward(self, h, c, g, **kwargs):
        if self.target in ["h", "both"]:
            h = self.in_bilinear(h.squeeze(0), g)
            h = self.tanh(h)
            h = self.out_bilinear(h, g).unsqueeze(0)
        if self.target in ["c", "both"]:
            c = self.in_bilinear(c.squeeze(0), g)
            c = self.tanh(c)
            c = self.out_bilinear(c, g).unsqueeze(0)
        return h, c

        
class PairClassifier(nn.Module):
    """
    A classifier for pairs of sequences.
    """

    def __init__(self, voab_1, vocab_2, embed_dim_1, embed_dim_2,
                 hidden_dim, class_dim, pair_encoder, n_layers,
                 n_classes, class_hidden_dim):
        
        super(PairClassifier, self).__init__()
        self.first_encoder = EncoderRNN(vocab_1, embed_dim_1, hidden_dim)
        self.second_encoder = EncoderRNN(vocab_2, embed_dim_2, hidden_dim)
        self.pair_encoder = pair_encoder
        
        self.classifier = nn.Sequential(nn.Linear(class_dim, class_hidden_dim), nn.Tanh())
        
        for i in range(n_layers):
            self.classifier.add(nn.Linear(class_hidden_dim, class_hidden_dim))
            self.classifier.add(nn.Tanh())
            
        self.classifier.add(nn.Linear(class_hidden_dim, n_classes))
        self.classifier.add(nn.LogSoftmax())
            

    def forward(self, input_1, input_2):
        h_1, hn_1, cn_1 = self.first_encoder(input1)
        h_2, hn_2, cn_2 = self.second_encoder(input2)
        encoded = self.pair_encoder(h_1, hn_1, cn_1, h_2, hn_2, cn_2)
        probs = self.classifier(encoded)
        return probs



class ConcatPairClassifier(PairClassifier):
    """
    A classifier for pairs of sequences that embeds and then concatenates them.
    """

    def __init__(self, voab_1, vocab_2, embed_dim_1, embed_dim_2,
                 hidden_dim, n_layers, n_classes, class_hidden_dim):

        #TODO add code for concatenation-based `pair_encoder`
        
        super(PairClassifier, self).__init__(voab_1, vocab_2, embed_dim_1, embed_dim_2,
                                             hidden_dim, class_dim, pair_encoder, n_layers,
                                             n_classes, class_hidden_dim)
        
        
    
