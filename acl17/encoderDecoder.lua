local nn = require "nn"
local rnn = require "rnn"
local cunn = require "cunn"
local dpnn = require "dpnn"
local nngraph = require "nngraph"
local optim = require "optim"
local beam = require "beamsearch"
local pastalog = require "pastalog"
local tnt = require "torchnet"
require "Replace"

local EncoderDecoder, parent = torch.class("EncoderDecoder", "nn.Module")

function EncoderDecoder:__init(inVocab, outVocab,
			       inEmbedSize, outEmbedSize,
			       hiddenSize, attention, dropout, gpu)

   parent.__init(self)
   self.output = {}
   -- store vocabularies
   self.inVocab = inVocab
   self.outVocab = outVocab
   local inVocabSize = vocabLength(inVocab["forward"])
   local outVocabSize = vocabLength(outVocab["forward"])
   print("inVocabSize: ", inVocabSize)
   print("outVocabSize: ", outVocabSize)

   -- Lookup tables
   self.inLookup = nn.LookupTableMaskZero(inVocabSize, inEmbedSize)
   self.outLookup = nn.LookupTableMaskZero(outVocabSize + 1, outEmbedSize)

   -- Encoder LSTM
   self.enc = nn.Sequential()
   self.enc:add(self.inLookup)
   if dropout > 0. then
      self.enc:add(nn.Dropout(dropout))
   end
   self.enc.lstmLayers = nn.SeqLSTM(inEmbedSize, hiddenSize)
   self.enc.lstmLayers:maskZero()
   self.enc:add(self.enc.lstmLayers)
   if dropout > 0. then
      self.enc:add(nn.Dropout(dropout))
   end
   self.enc:add(nn.Transpose({1, 2}))


   -- Decoder LSTM

   self.dec = nn.Sequential()
   self.dec:add(self.outLookup)
   if dropout > 0 then
      self.dec:add(nn.Dropout(dropout))
   end
   self.dec.lstmLayers = {}
   table.insert(self.dec.lstmLayers, nn.SeqLSTM(outEmbedSize, hiddenSize))
   table.insert(self.dec.lstmLayers, nn.SeqLSTM(hiddenSize, hiddenSize))

   for i = 1,#self.dec.lstmLayers do
      self.dec.lstmLayers[i]:maskZero()
      self.dec:add(self.dec.lstmLayers[i])
   end
   if dropout > 0 then
      self.dec:add(nn.Dropout(dropout))
   end
   self.dec:add(nn.SplitTable(1))

   -- Attention and Generator
   self.attention = attention
   --self.attention = nn.Recursor(Attention(hiddenSize, hiddenSize, hiddenSize))
   if self.attention then
      print("Using attention.")
      local inputs = {nn.Identity()(), nn.Identity()()}
      local cxt = Attention(inputs, hiddenSize, false)

      self.generator = nn.Sequential()
	 :add(nn.ZipTableOneToMany())
	 :add(nn.Sequencer(nn.gModule(
			      inputs,
			      {nn.Unsqueeze(1)(
				  nn.MaskZero(
				     nn.LogSoftMax(),
				     1
				  )(
				     nn.MaskZero(
					nn.Linear(hiddenSize,
						  outVocabSize),
					1)(cxt)))})))
	 :add(nn.JoinTable(1))

   else
      print("Using no attention.")
      self.generator = nn.Sequential()
	 :add(nn.SelectTable(2))
	 :add(nn.Sequencer(
		 nn.Sequential()
		    :add(nn.MaskZero(
			    nn.Linear(hiddenSize,
				      outVocabSize),
			    1))
		    :add(nn.MaskZero(
			    nn.LogSoftMax(),
			    1))
		    :add(nn.Unsqueeze(1))))
	 :add(nn.JoinTable(1))
   end

   -- Loss function
   --self.criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1))
   self.splitter = nn.SplitTable(1)


   -- run on GPU
   if gpu then
      self.enc:cuda()
      self.dec:cuda()
      self.generator:cuda()
      self.splitter:cuda()
      --self.criterion:cuda()
   end

   -- TODO: add generator weights
   -- get weights and weight gradients
   local encWeights, encGradWeights = self.enc:parameters()
   local decWeights, decGradWeights = self.dec:parameters()
   local genWeights, genGradWeights = self.generator:parameters()
   local weights = append(append(encWeights, decWeights), genWeights)
   local gradWeights = append(append(encGradWeights, decGradWeights), genGradWeights)
   self.weights = nn.Module.flatten(weights)
   self.gradWeights = nn.Module.flatten(gradWeights)

   self.paramFun = function()
      return self.weights, self.gradWeights
   end

   -- beam search
   self.beamSearcher = BeamSearch(10, 300, self.outVocab["reverse"]["('shift', '-')"],
				  self.outVocab["reverse"]["<END>"],
				  self.outVocab["reverse"]["<END>"])
   self.startSymbol = self.outVocab["reverse"]["('shift', '-')"]
   self.endSymbol = self.outVocab["reverse"]["('right-arc', 'root')"]
   self.stateFunc = function()
      out = {}
      for i = 1,#self.dec.lstmLayers do
	 table.insert(out, {self.dec.lstmLayers[i].output[1]:clone(), self.dec.lstmLayers[i].cell[1]:clone()})
      end
      return out
   end

   self.copyFunc = function(states)
      for i, state in pairs(states) do
	 self.dec.lstmLayers[i].userPrevOutput = state[1]
	 self.dec.lstmLayers[i].userPrevCell = state[2]
      end
   end

end

function append(t1, t2)
   local out = {}
   for k, v in pairs(t1)
   do
      table.insert(out, v)
   end
   for k, v in pairs(t2)
   do
      table.insert(out, v)
   end
   return out
end

function EncoderDecoder:zeroGradParameters()
   self.gradWeights:zero()
end

function EncoderDecoder:greedy(input, maxLength)
   local i = 1
   local symbol = self.startSymbol
   local sequence = {symbol}
   local encoding = self.enc:forward(input)

   self.dec:forget()
   self.generator:forget()
   forwardConnect(self.enc, self.dec, input:size(1))
   self.dec:remember()
   self.generator:remember()
   while (symbol ~= self.endSymbol) and (i < maxLength) do
      symbol = torch.Tensor({symbol}):reshape(1, 1)
      local decOut = self.dec:forward(symbol)
      local predOut = self.generator:forward({encoding, decOut})
      local top, idx = torch.topk(predOut, 1, 3, true)
      symbol = idx[{1, 1, 1}]
      table.insert(sequence, symbol)
      i = i+1
   end
   return sequence
end

function EncoderDecoder:forward(input)
   local encInSeq, decInSeq = unpack(input)
   local inSeqLen = encInSeq:size(1)
   local encOut = self.enc:forward(encInSeq)
   forwardConnect(self.enc, self.dec, inSeqLen)
   local decOut = self.dec:forward(decInSeq)
   self.output = self.generator:forward({encOut, decOut})
   return self.output
end

function EncoderDecoder:backward(input, gradOutput)
   local encInSeq, decInSeq = unpack(input)
   local encOut = self.enc.output
   local decOut = self.dec.output
   local grads = self.generator:backward({encOut, decOut}, gradOutput)
   local encGrad, decGrad = unpack(grads)
   local decInGrad = self.dec:backward(decInSeq, decGrad)
   backwardConnect(self.enc, self.dec)
   local encInGrad = self.enc:backward(encInSeq, encGrad)
   self.gradInput = {encInGrad, decInGrad}
   return self.gradInput
end


function EncoderDecoder:eval(dataIterator)
   local numSamples = {}
   local errs = {}

   for input, output, SO in dataIterator do
      table.insert(numSamples, input:size(2))
      local err, _, _ = self:forward(input, output, SO)
      table.insert(errs, math.exp(err/output:size(1)))
   end

   numSamples = torch.Tensor(numSamples)
   errs = torch.Tensor(errs)

   return torch.cmul(numSamples, errs):sum()/numSamples:sum()
end

function EncoderDecoder:generate(input)
   local encOut = self.enc:forward(input)
   forwardConnect(self.enc, self.dec, seqLen)
   out = self.beamSearcher:search(self.dec, self.stateFunc, self.copyFunc)

   return out["y"]
end

function EncoderDecoder:decodeVocab(input, inp)

   local decVocab

   if inp then
      decVocab = self.inVocab["forward"]
   else
      decVocab = self.outVocab["forward"]
   end

   local res = {}
   for i, id in pairs(input) do
      table.insert(res, decVocab[tostring(id)])
   end

   return res
end

-- Forward coupling: Copy encoder cell and output to decoder LSTM
function forwardConnect(enc, dec, seqLen)
   dec.lstmLayers[1].userPrevOutput = enc.lstmLayers.output[seqLen]
   dec.lstmLayers[1].userPrevCell = enc.lstmLayers.cell[seqLen]
end

function backwardConnect(enc, dec)
   enc.lstmLayers.userNextGradCell = dec.lstmLayers[1].userGradPrevCell
   enc.lstmLayers.gradPrevOutput = dec.lstmLayers[1].userGradPrevOutput
end

function vocabLength(vocab)
   local keyset={}

   for k,v in pairs(vocab) do
      table.insert(keyset, k)
   end
   return torch.Tensor(keyset):max()
end

function Attention(inputs, hiddenSize, return_weights)
   -- 2D tensor target_t (batch_l x rnn_size) and
   -- 3D tensor for context (batch_l x source_l x rnn_size)

   local context, target = unpack(inputs)
   local target_t = nn.LinearNoBias(hiddenSize, hiddenSize)(target)
   -- get attention

   local scores = nn.MM()({context, nn.Unsqueeze(3)(target_t)}) -- batch_l x source_l x 1
   scores = nn.Squeeze(3)(scores)
   scores = nn.Replace(0, -math.huge)(scores)
   local softmax_attn = nn.SoftMax()
   if return_weight then
      local logattn = nn.LogSoftMax()(scores)
      logattn = nn.Replace("nan", 0)(logattn)
   end
   attn = softmax_attn(scores)
   attn = nn.Replace("nan", 0)(attn)
   attn = nn.Unsqueeze(2)(attn) -- batch_l x  1 x source_l

   -- apply attention to context
   local context_combined = nn.MM()({attn, context}) -- batch_l x 1 x rnn_size
   context_combined = nn.Squeeze(2)(context_combined) -- batch_l x rnn_size
   local context_output
   context_combined = nn.JoinTable(2)({context_combined, target}) -- batch_l x rnn_size*2
   context_output = nn.Tanh()(nn.LinearNoBias(hiddenSize*2, hiddenSize)(context_combined))
   if return_weights then
      return context_output, logattn
   else
      return context_output
   end
end
