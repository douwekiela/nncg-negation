local nn = require "nn"
local rnn = require "rnn"
local optim = require "optim"
local nng = require "nngraph"
local cutorch = require "cutorch"
local tnt = require "torchnet"
local pastalog = require "pastalog"
package.path = "./lib/?.lua"..package.path
require "lib.encoderDecoder"
require "lib.gatedEncoderDecoder"
require "lib.dataUtils"

function generateData(eval, n)
   local data = {}
   for i=1,n do
      local min = torch.random(3, 12)
      local max = torch.random(min, 12)
      local input = torch.range(min, max, 1)
      -- local input = eval and torch.range(max, min, -1) or torch.range(min, max, 1)
      local output = torch.cat({torch.Tensor{1}, input, torch.Tensor{2}})
      data[i] = {input = input, output = output, context = torch.rand(300)}
   end
   return data
end

function main(opt)
   print(opt)
   
   local cuda
   if opt.gpu == -1 then
      cuda = false
   else
      cutorch.setDevice(opt.gpu)
      cuda = true
   end
   
   local vocab = {forward = {"<START>", "<END>", "0",
			     "1", "2", "3", "4",
			     "5", "6", "7", "8",
			      "9"},
		  reverse = {["<START>"]= 1, ["<END>"]= 2, ["0"]= 3,
		     ["1"]=4, ["2"]= 5, ["3"]= 6, ["4"]= 7,
		     ["5"]= 8, ["6"]= 9, ["7"]= 10, ["8"]= 11,
		     ["9"]= 12}}
   
   local model = EncoderDecoder(vocab, vocab, opt.dimHid,
				opt.attn, opt.dropout,
				cuda, ".", true, opt.dimEmb,
				opt.dimEmb)

   local criterion = nn.SequencerCriterion(
      nn.MaskZeroCriterion(
	 nn.ClassNLLCriterion(),
	 1))
   
   if cuda then
      criterion:cuda()
   end
  
   local trainData = loadDataset(generateData(false, opt.nTrain),
				 opt.batchSize, cuda, false)
   local engine = tnt.OptimEngine()
   local loss = tnt.AverageValueMeter()
   if cuda then
      engine.hooks.onSample = function(state)
	 state.sample.input[1] = state.sample.input[1]:cuda()
	 state.sample.input[2] = state.sample.input[2]:cuda()
	 state.sample.target = state.sample.target:cuda()
      end
   end
   
   engine.hooks.onForwardCriterion = function(state)
      print("Forward criterion.")
      local err = state.criterion.output
      local seqLen = state.sample.target:size(1)
      print("loss:", err)
      print("average loss:", err/seqLen)
      loss:add(err/seqLen)
      print("approx. perplexity:", math.exp(err/seqLen))
      pastalog(opt.modelName, "average NLL.", err/seqLen, state.t, "http://localhost:9000/data")
   end

   engine.hooks.onEndEpoch = function(state)
      print("End of Epoch ", state.epoch)
      loss:reset()
      if opt.overwrite then
         torch.save(opt.saveDir.."/"..opt.modelName..".t7",
		    state.network, "binary")
      elseif (state.epoch % 100 == 0) then
	 torch.save(opt.saveDir.."/"..opt.modelName.."-"..state.epoch..".t7",
		    state.network, "binary")
      end
   end
   
   engine:train{
      network = model,
      iterator = trainData,
      criterion = criterion,
      optimMethod = optim.adam,
      config = {
	 learningRate = opt.LR 
      },
      paramFun = model.paramFun,
      maxepoch = opt.nEpochs
   }

   local evalData = generateData(true, opt.nEval)
   model:regenerateBeam(opt.beamSize, opt.maxLength)
   for i=1,#evalData do
      local example = evalData[i]
      local input = example["input"]
      local gate = torch.rand(300)
      local resInput = torch.Tensor(input:size(1)):copy(input):resize(input:size(1), 1):cuda()
      local resGate = torch.Tensor(gate:size(1)):copy(gate):resize(1, gate:size(1)):cuda()
      local printOutput = model:decodeVocab(torch.totable(example["output"]), true)
      local printInput = model:decodeVocab(torch.totable(example["input"]), true)
      local greedyModelOutput = model:greedy(resInput, resGate, 50)
      print("input:", example["input"])
      print("output:", example["output"]) 
      print("greedyModelOutput:", greedyModelOutput)
      local beamModelOutput = model:generate(resInput, resGate, opt.beamSize, opt.maxLen) 
      print("beamModeloutput:", beamModelOutput)
   end
end

local cmd = torch.CmdLine()

cmd:text()
cmd:text("Encoder-decoder negation training.")
cmd:text()
cmd:text("Options:")
---------------------- Model Options -------------------------------
cmd:option("-dimEmb", 15, "Character embedding vector dimension.")
cmd:option("-dimHid", 20, "Hidden vector dimension.")
cmd:option("-dropout", 0, "Dropout rate.")
cmd:option("-attn", false, "Use attentive decoder.")
-------------------- Training Options -----------------------------
cmd:option("-nTrain", 2000, "Number of training examples epochs.")
cmd:option("-nEpochs", 10, "Number of total epochs.")
cmd:option("-LR", 0.001, "Learning rate.")
cmd:option("-batchSize", 128, "Mini-batch size.")
cmd:option("-gpu", -1, "GPU device to use. -1 for CPU.")
cmd:option("-LR", 0.001, "Learning rate.")
-------------------- Evaluation Options ---------------------------
cmd:option("-nEval", 100, "Number of eval examples.")
cmd:option("-beamSize", 10, "Beam size.")
cmd:option("-maxLen", 100, "Maximum sentence length for beam search.")
-------------------- General Options ------------------------------
cmd:option("-modelName", "testCopy", "Name for your model files.")
cmd:option("-saveDir", "/local/filespace/am2156/nncg/models/",
	   "Directory to save models to.")
-------------------------------------------------------------------
local opt = cmd:parse(arg)
main(opt)
