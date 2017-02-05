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

   local criterion = nn.SequencerCriterion(
      nn.MaskZeroCriterion(
	 nn.ClassNLLCriterion(),
	 1))
   if cuda then
      criterion:cuda()
   end

   local engine = tnt.OptimEngine()
   local loss = tnt.AverageValueMeter()
   local losstab = {}

   if cuda then
      engine.hooks.onSample = function(state)
	 state.sample.input[1] = state.sample.input[1]:cuda()
	 state.sample.input[2] = state.sample.input[2]:cuda()
	 state.sample.input[3] = state.sample.input[3]:cuda()
	 state.sample.target = state.sample.target:cuda()
      end
   end

   engine.hooks.onStart = function(state)
      print("Starting training.")
      loss:reset()
   end
   
   engine.hooks.onForwardCriterion = function(state)
      --print("Forward criterion.")
      local err = state.criterion.output
      local seqLen = state.sample.target:size(1)
      io.stdout:flush()
      io.stdout:write("Epoch: ", state.epoch, ", avg. loss: ", err/seqLen, ", ppl.: ", math.exp(err/seqLen), "\r")
      loss:add(err/seqLen)
      pastalog(opt.modelName, "log-perplexity", err/seqLen, state.t, "http://localhost:9000/data")
   end

   engine.hooks.onEndEpoch = function(state)
      table.insert(losstab, loss:value())
      loss:reset()
      if opt.overwrite then
         torch.save(modelDir.."/"..opt.modelName..".t7",
		    state.network, "binary")
      elseif (state.epoch % 100 == 0) then
	 torch.save(modelDir.."/"..opt.modelName.."-"..state.epoch..".t7",
		    state.network, "binary")
      end
   end

   local outData = loadFromJSON(opt.trainFile, false)
   local trainData = loadDataset(outData, opt.batchSize, cuda, false)

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
