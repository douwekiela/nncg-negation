local cutorch = require "cutorch"
local nn = require "nn"
local rnn = require "rnn"
local cunn = require "cunn"
local dpnn = require "dpnn"
local nngraph = require "nngraph"
local npy4th = require 'npy4th'
local optim = require "optim"
local beam = require "lib.beamsearch"
local tnt = require "torchnet"
require "lib.Replace"

local GatedEncoderDecoder, parent = torch.class("GatedEncoderDecoder", "nn.Module")

function GatedEncoderDecoder:__init(inVocab, outVocab,
				    hiddenSize, attention,
				    inDropout, outDropout,
				    gpu, embPath,
				    trainEmb,
				    inEmbedSize,
				    outEmbedSize,
				    gateHiddenSize)

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
   self.trainEmb = trainEmb
   self.inLookup = nn.LookupTableMaskZero(inVocabSize, inEmbedSize)
   self.outLookup = nn.LookupTableMaskZero(outVocabSize, outEmbedSize)
      
   if embPath ~= "." then
      self:loadEmbedding(embPath, gpu)
      if not self.trainEmb then
	 self.inLookup.accGradParameters = function(self, i, o, s) end
	 self.outLookup.accGradParameters = function(self, i, o, s) end
      end
   end
   
   -- gatedEncoder LSTM
   self.enc = nn.Sequential()
   self.enc:add(self.inLookup)
   if inDropout > 0. then
      self.enc:add(nn.Dropout(inDropout))
   end
   self.enc.lstmLayers = nn.SeqLSTM(inEmbedSize, hiddenSize)
   self.enc.lstmLayers:maskZero()
   self.enc:add(self.enc.lstmLayers)
   -- if dropout > 0. then
   --    self.enc:add(nn.Dropout(dropout))
   -- end
   self.enc:add(nn.Transpose({1, 2}))

   -- Defining gated network

   local inhid, ingate = nn.Identity()(), nn.Identity()()
   local gatehid = nn.Tanh()(nn.Bilinear(hiddenSize, inEmbedSize, gateHiddenSize)({inhid, ingate}))
   local gateOut = nn.Bilinear(gateHiddenSize, inEmbedSize, hiddenSize)({gatehid, ingate})
   self.gatedEncoder = nn.gModule({inhid, ingate}, {gateOut})
   
   -- Decoder LSTM

   self.dec = nn.Sequential()
   self.dec:add(self.outLookup)
   if outDropout > 0 then
      self.dec:add(nn.Dropout(outDropout))
   end
   self.dec.lstmLayers = {}
   table.insert(self.dec.lstmLayers, nn.SeqLSTM(outEmbedSize, hiddenSize))
   table.insert(self.dec.lstmLayers, nn.SeqLSTM(hiddenSize, hiddenSize))

   for i = 1,#self.dec.lstmLayers do
      self.dec.lstmLayers[i]:maskZero()
      self.dec:add(self.dec.lstmLayers[i])
   end
   -- if dropout > 0 then
   --    self.dec:add(nn.Dropout(dropout))
   -- end
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
      self.gatedEncoder:cuda()
      self.generator:cuda()
      self.splitter:cuda()
      --self.criterion:cuda()
   end

   -- get weights and weight gradients
   local gateWeights, gateGradWeights = self.gatedEncoder:parameters()
   local encWeights, encGradWeights = self.enc:parameters()
   local decWeights, decGradWeights = self.dec:parameters()
   local genWeights, genGradWeights = self.generator:parameters()
   local weights = append(append(append(encWeights, decWeights), genWeights), gateWeights)
   local gradWeights = append(append(append(encGradWeights, decGradWeights), genGradWeights), gateGradWeights)
   self.weights = nn.Module.flatten(weights)
   self.gradWeights = nn.Module.flatten(gradWeights)

   self.paramFun = function()
      return self.weights, self.gradWeights
   end

   -- beam search
   self.startSymbol = self.outVocab["reverse"]["<START>"]
   self.endSymbol = self.outVocab["reverse"]["<END>"]
   self.beamSearcher = BeamSearch(10, 300, self.startSymbol,
				  self.endSymbol, self.startSymbol)
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

function GatedEncoderDecoder:regenerateBeam(beamSize, maxLength)
   self.beamSearcher = BeamSearch(beamSize, maxLength, self.startSymbol,
				  self.endSymbol, self.startSymbol)
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

function GatedEncoderDecoder:zeroGradParameters()
   self.gradWeights:zero()
end
function GatedEncoderDecoder:forgetStates()
   for i, layer in pairs(self.dec.lstmLayers) do
      layer._remember = 'neither'
   end
end
function GatedEncoderDecoder:greedy(input, gate, maxLength)
   local i = 1
   local symbol = self.startSymbol
   local sequence = {symbol}
   local encoding = self.enc:forward(input)

   self.dec:forget()
   self.generator:forget()
   gatedForwardConnect(self.enc, self.dec, self.gatedEncoder, gate, input:size(1))
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
   self.dec:forget()
   self.generator:forget()
   self:forgetStates()

   return sequence
end

function GatedEncoderDecoder:forward(input)
   local encInSeq, decInSeq, gate = unpack(input)
   local inSeqLen = encInSeq:size(1)
   local encOut = self.enc:forward(encInSeq)
   self.dec:forget()
   self.generator:forget()
   gatedForwardConnect(self.enc, self.dec, self.gatedEncoder, gate, inSeqLen)
   self.dec:remember()
   self.generator:remember()
   local decOut = self.dec:forward(decInSeq)
   self.output = self.generator:forward({encOut, decOut})
   return self.output
end

function GatedEncoderDecoder:backward(input, gradOutput)
   local encInSeq, decInSeq, gate = unpack(input)
   local encOut = self.enc.output
   local decOut = self.dec.output
   local grads = self.generator:backward({encOut, decOut}, gradOutput)
   local encGrad, decGrad = unpack(grads)
   local decInGrad = self.dec:backward(decInSeq, decGrad)
   gatedBackwardConnect(self.enc, self.dec, self.gatedEncoder, gate)
   local encInGrad = self.enc:backward(encInSeq, encGrad)
   self.gradInput = {encInGrad, decInGrad}
   return self.gradInput
end


function GatedEncoderDecoder:eval(dataIterator)
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

function GatedEncoderDecoder:generate(input, gate, beamSize, maxLen)
   local tabCopy = function(tab)
      local copy = {}
      for i, elt in pairs(tab) do
	 copy[i] = elt
      end
      return copy
   end
   local ij = function(k, n)
      local i, j
      j = torch.fmod(torch.Tensor{k}, n)[1]
      j = (j==0) and n or j
      i = (k-j)/n + 1
      return i, j
   end
   local dimH = self.dec.lstmLayers[1].hiddensize
   local numLayers = #self.dec.lstmLayers
   local encOut = self.enc:forward(input)
   self.dec:forget()
   self.generator:forget()
   gatedForwardConnect(self.enc, self.dec, self.gatedEncoder, gate, input:size(1))
   self.dec:remember()
   self.generator:remember()
   local beam = {}
   local complete = {}
   local pruneFactor = torch.log(0)

   local decOut = self.dec:forward(torch.Tensor{self.startSymbol}:resize(1, 1))
   local scores = self.generator:forward({encOut, decOut})
   local topScores, topIds = scores:sort(3,true)
   local dimWord = scores:size(3)
   

   for i=1,beamSize do
      local id, score =  topIds[{1, 1, i}], topScores[{1, 1, i}]
      local state = {}
      for j=1,numLayers do
	 state[j] = {cell = torch.CudaTensor(1, dimH):copy(self.dec.lstmLayers[j].cell),
		     output = torch.CudaTensor(1, dimH):copy(self.dec.lstmLayers[j].output)}
      end
      if id == self.endSymbol then
	 table.insert(complete, {seq = {self.startSymbol, id},
				 score = score}) 
	 pruneFactor = score
      else
	 table.insert(beam, {seq = {self.startSymbol, id},
			     state = state,
			     score = score})
      end
   end
   
   local counter = 1
   while (#beam > 0) and (counter<maxLen) do
      -- Copy states
      local beamSymbols, beamStates = {}, {}
      
      for l=1,numLayers do
	 beamStates[l] = {cell = {}, output = {}}
      end
      
      for j=1,#beam do
	 beamSymbols[j] = beam[j]["seq"][#beam[j]["seq"]]
	 for k=1,numLayers do
	    beamStates[k]["cell"][j] = beam[j]["state"][k]["cell"]
	    beamStates[k]["output"][j] = beam[j]["state"][k]["output"]
	 end
      end
      
      beamSymbols = torch.Tensor(beamSymbols):resize(1, #beam)
      for m=1,numLayers do
	 beamStates[m]["cell"] = torch.cat(beamStates[m]["cell"], 1):resize(#beam, dimH)
	 beamStates[m]["output"] = torch.cat(beamStates[m]["output"], 1):resize(#beam, dimH)
      end
      
      -- Calculate probabilities
      self.dec:forget()
      self.generator:forget()
      self:copyDecState(beamStates)
      
      
      local decOut = self.dec:forward(beamSymbols)
      local scores = self.generator:forward({encOut, decOut}):resize(#beam, dimWord)

      local dimBeam, dimWord = scores:size(1), scores:size(2)
      assert(dimBeam == #beam, "Batch size must equal beam size.")
      local flatScores = scores:view(scores:nElement())
      local topScores, topIds = flatScores:sort(1, true)
      local outStates = {}

      for n=1,#beam do
	 local layerStates = {}
	 for p=1,numLayers do
	    layerStates[p] = {cell = torch.CudaTensor(1, dimH):copy(self.dec.lstmLayers[p].cell[{{}, n, {}}]),
			      output = torch.CudaTensor(1, dimH):copy(self.dec.lstmLayers[p].output[{{}, n, {}}])}
	 end
	 outStates[n] = layerStates
      end
      
      -- Update beam
      local newBeam = {}
      for id=1,math.min(beamSize, #beam) do
	 local score = topScores[{id}]
	 local i, j = ij(topIds[id], dimWord)

	 if score>=pruneFactor then
	    local seq = tabCopy(beam[i]["seq"])
	    table.insert(seq, j)
	    if j==self.endSymbol then 
	       table.insert(complete, {seq = seq,
				       score = beam[i]["score"] + score}) 
	       pruneFactor = score
	    else
	       table.insert(newBeam, {seq = seq,
				      state = outStates[i],
				      score = beam[i]["score"]+ score})
	    end
	 else
	    break
	 end
      end
      beam = newBeam
      counter = counter + 1
   end

   return complete[#complete]["seq"]
end

function GatedEncoderDecoder:copyDecState(states)
   for i, state in pairs(states) do
      self.dec.lstmLayers[i].userPrevOutput = state["output"]
      self.dec.lstmLayers[i].userPrevCell = state["cell"]
   end
end

function GatedEncoderDecoder:decodeVocab(input, inp)

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

function GatedEncoderDecoder:loadEmbedding(filename, gpu)
   local emb = npy4th.loadnpy(filename):double()
   emb = torch.cat({torch.zeros(1, self.inLookup.weight:size(2)),
		    torch.rand(2, self.inLookup.weight:size(2)),
		    emb}, 1)
   -- if gpu then
   --    emb = emb:cuda()
   -- end
   print("emb: ", type(emb), emb:size())
   self.inLookup.weight = torch.Tensor(emb:size(1), emb:size(2)):copy(emb):double()
   self.outLookup.weight = torch.Tensor(emb:size(1), emb:size(2)):copy(emb):double()
end

-- Forward coupling: Copy encoder cell and output to decoder LSTM
function gatedForwardConnect(enc, dec, gateEnc, gate, seqLen)
   dec.lstmLayers[1].userPrevOutput =  gateEnc:forward({enc.lstmLayers.output[seqLen], gate})
   dec.lstmLayers[1].userPrevCell =  gateEnc:forward({enc.lstmLayers.cell[seqLen], gate})
end

function gatedBackwardConnect(enc, dec, gateEnc, gate)
   local gradOutput, _ = unpack(gateEnc:backward({enc.lstmLayers.output[seqLen], gate},
   				   dec.lstmLayers[1].userGradPrevOutput))
   local gradCell, _ = unpack(gateEnc:backward({enc.lstmLayers.cell[seqLen], gate},
				 dec.lstmLayers[1].userGradPrevCell))
   enc.lstmLayers.userNextGradCell = gradCell
   enc.lstmLayers.gradPrevOutput = gradOutput
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
