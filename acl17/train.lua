local js = require "cjson"
local nn = require "nn"
local rnn = require "rnn"
local optim = require "optim"
local nng = require "nngraph"
local cutorch = require "cutorch"
local tnt = require "torchnet"
local sys = require "sys"
local pastalog = require "pastalog"
package.path = "./lib/?.lua"..package.path
require "lib.encoderDecoder"
require "lib.gatedEncoderDecoder"
require "lib.dataUtils"

function main(opt)
   print(opt)
   local cuda
   if opt.gpu == -1 then
      cuda = false
   else
      print(type(opt.gpu))
      cutorch.setDevice(opt.gpu)
      cuda = true
   end 

   local wordVocab = js.decode(io.open(opt.wordVocab, "r"):read("*all"))
   local modelDir
   local model
   if opt.loadModel == "none" then
      modelDir = opt.saveDir--..os.date():gsub("%s+", "-")
      --sys.execute("mkdir "..modelDir)
      if opt.gated then
	    model = GatedEncoderDecoder(wordVocab,
					wordVocab,
					opt.dimHid,
					opt.attention,
					opt.inDropout, opt.outDropout,
					cuda, opt.embPath,
					opt.trainEmb,
					opt.dimWord,
					opt.dimWord,
					opt.dimGateHid)
	 else
	    model = EncoderDecoder(wordVocab,
				   wordVocab,
				   opt.dimHid,
				   opt.attention,
				   opt.inDropout, opt.outDropout,
				   cuda, opt.embPath,
				   opt.trainEmb,
				   opt.dimWord,
				   opt.dimWord)
	 end
   else
      modelDir = opt.loadModel
      print("Loading model...")
      model = torch.load(opt.loadModel.."/"..opt.modelName..".t7", "binary")
   end

   local criterion = nn.ClassNLLCriterion()
   local lmCriterion = nn.SequencerCriterion(
      nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1)
   )
   local rankCriterion = nn.MarginRankingCriterion(opt.margin) 
   local jointCriterion = nn.ParallelCriterion()
   jointCriterion:add(lmCriterion, opt.lmWeight):add(rankCriterion, 1-opt.lmWeight)
   if cuda then
      criterion:cuda()
      rankCriterion:cuda()
      jointCriterion:cuda()
   end

   function modelScorerForward(input, model)
      local In, Out, Con = unpack(input.input)
      local Target = input.target
      local modProbs = model:forward({In, Out, Con})
      local scores, target = {}, {}
      for i=1,Target:size(2) do
	 local probs, targs = modProbs[{{}, i, {}}], Target[{{}, i}] 
	 local indLast = torch.find(targs, 0)
	 indLast = (#indLast > 0) and indLast[1] - 1  or targs:size(1)
	 local probs, targs = probs[{{1, indLast}, {}}], targs[{{1, indLast}}]
	 local logprob = 0
	 for t=1,indLast do
	    logprob = logprob - criterion:forward(probs[{t, {}}], targs[{t}])
	 end
	 scores[i] = logprob
      end
      -- local copyProbs = torch.Tensor(modProbs:size()):copy(modProbs)
      scores = torch.Tensor(scores)
      -- local encOutCopy = torch.Tensor(encOut:size()):copy(encOut)
      -- local decOutCopy = {}
      -- for i=1,#decOut do 
      -- 	 decOutCopy[i] = cuda and torch.CudaTensor(decOut[i]:size()):copy(decOut[i]) or torch.CudaTensor(decOut[i]:size()):copy(decOut[i])
      -- end
      if cuda then
	 scores = scores:cuda()
	 -- copyProbs = copyProbs:cuda()
	 -- encOutCopy = encOutCopy:cuda()
      end
      return scores, modProbs
   end
   
   function modelScorerBackward(input, model, gradOut, initGrad)
      local modelLogLik = model.output
      local In, Out, Con = unpack(input.input)
      local Target = input.target
      local scoreGrads = {}
      local maxLen = 0
      for i=1,Target:size(2) do
	 local logLiks, targs = modelLogLik[{{}, i, {}}], Target[{{}, i}]
	 local indLast = torch.find(targs, 0)
	 indLast = (#indLast > 0) and indLast[1] - 1 or targs:size(1)
	 local probs, targs = logLiks[{{1, indLast}, {}}], targs[{{1, indLast}}]
	 local scoreGradBatch = {}
	 for t=1,indLast do
	    local scoreGradTime = -criterion:backward(probs[{t, {}}], targs[{t}])
	    local scoreGradTime = scoreGradTime:reshape(1, 1, scoreGradTime:size(1))
	    scoreGradBatch[t] = scoreGradTime
	 end
	 scoreGradBatch = torch.cat(scoreGradBatch, 1)
	 scoreGradBatch = gradOut[{i}]*scoreGradBatch
	 local indDiff = Target:size(1) - indLast
	 if indDiff > 0 then
	    local zeros = torch.zeros(indDiff, 1, probs:size(2))
	    scoreGradBatch = torch.cat({scoreGradBatch, cuda and zeros:cuda() or zeros}, 1)
	 end
	 
	 scoreGrads[i] = scoreGradBatch
      end
      scoreGrads = torch.cat(scoreGrads, 2)
      if cuda then
	 scoreGrads = scoreGrads:cuda()
      end
      if initGrad then
	 scoreGrads = initGrad + scoreGrads
      end
      model:backward({In, Out, Con}, scoreGrads)
      end
      
      local loss = tnt.AverageValueMeter()
      local losstab = {}

      local outData = loadFromJSON(opt.trainFile, false)
      local trainData = loadDataset(outData, opt.batchSize, cuda, false)
      
      print("Starting training.")
      loss:reset()
      params, gradParams = model.paramFun()

      local function feval()
      return rankCriterion.output, gradParams
   end
   config = {
      learningRate = opt.LR 
   }
   local goldModel, noiseModel = model, model:customClone()
   t = 1
   for epoch=1,opt.nEpochs do
      for sample in trainData() do
	 ------- Sample --------------------------
	 local gold, noise = unpack(sample.input)
	 --------Forward network -------------------------
	 local goldLogLik, goldProbs = modelScorerForward(gold, goldModel)
	 local noiseLogLik, _ = modelScorerForward(noise, noiseModel)
	 ------- Forward criterion ---------------
	 local rankInput, rankTarget = {goldLogLik, noiseLogLik}, sample.target
	 local lmInput, lmTarget = goldProbs, gold.target
	 local jointInput, jointTarget = {lmInput, rankInput}, {lmTarget, rankTarget}
	 local err = jointCriterion:forward(jointInput, jointTarget)
	 
	 -------log results -----------------------
	 local seqLen = sample.target:size(1)
	 io.stdout:flush()
	 io.stdout:write("Epoch: ", epoch, ", avg. loss: ", err, ", ppl.: ", math.exp(err/seqLen), "\r")
	 loss:add(err/seqLen)
	 pastalog(opt.modelName, "log-perplexity", err/seqLen, t, "http://localhost:9000/data")

	 model:zeroGradParameters()
	 
	 ------- Backward criterion -------------
	 local jointGrad = jointCriterion:backward(jointInput, jointTarget)
	 local gradLM, gradRank = unpack(jointGrad)

	 --------Backward network ---------------
	 modelScorerBackward(gold, goldModel, gradRank[1], gradLM)
	 modelScorerBackward(noise, noiseModel, gradRank[2])

	 optim.adam(feval, params, config, {})
	 --pastalog(opt.modelName, "gradNorm", model.gradWeights:norm(), t, "http://localhost:9000/data")
	 t = t + 1
      end
      
      table.insert(losstab, loss:value())
      loss:reset()
      if opt.overwrite then
         torch.save(modelDir.."/"..opt.modelName..".t7",
		    model, "binary")
      elseif (epoch % 100 == 0) then
	 torch.save(modelDir.."/"..opt.modelName.."-"..epoch..".t7",
		    model, "binary")
      end    
   end


end

local cmd = torch.CmdLine()

cmd:text()
cmd:text("Encoder-decoder negation training.")
cmd:text()
cmd:text("Options:")
-------------------- Model options ---------------------------------
cmd:option("-dimWord", 300, "Word embedding dimension.")
cmd:option("-dimHid", 150, "Hidden vector dimension.")
cmd:option("-dimGateHid", 50, "Gated encoder hidden vector dimension.")
cmd:option("-inDropout", 0, "Dropout rate.")
cmd:option("-outDropout", 0, "Dropout rate.")
cmd:option("-attn", false, "Use attentive decoder.")
cmd:option("-trainEmb", false, "Tune word embeddings.")
cmd:option("-gated", false, "Use gated encoder-decoder.")
-------------------- Training options ------------------------------
cmd:option("-nEpochs", 10, "Number of total epochs.")
cmd:option("-margin", 0.7, "Margin for log-probability ranking loss.")
cmd:option("-lmWeight", 0.7, "Weight for language modelling loss.")
cmd:option("-batchSize", 128, "Mini-batch size.")
cmd:option("-gpu", -1, "GPU device to use. -1 for CPU.")
------------------- Optimization options ---------------------------
cmd:option("-LR", 0.001, "Learning rate.")
------------------- Data options -----------------------------------
cmd:option("-trainFile", "./data/train.json", "Training data JSON file")
cmd:option("-embPath", ".", "Pretrained embedding JSON file")
cmd:option("-modelName", "test", "Name for your model files.")
cmd:option("-overwrite", false, "Overwrite model files.")
cmd:option("-loadModel", "none", "Directory of model to load.")
cmd:option("-saveDir", "/local/filespace/am2156/nncg/models/", "Directory to save models to.")
cmd:option("-wordVocab", "./data/vocab.json", "Word vocabulary JSON file")

local opt = cmd:parse(arg)
main(opt)
