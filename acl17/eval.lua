local nn = require "nn"
local cunn = require "cunn"
local rnn = require "rnn"
local optim = require "optim"
local nng = require "nngraph"
local cutorch = require "cutorch"
local tnt = require "torchnet"
local js = require "cjson"
package.path = "./lib/?.lua"..package.path
require "lib.encoderDecoder"
require "lib.gatedEncoderDecoder"
require "lib.dataUtils"
require "torchx"


function main(opt)
   local cuda
   if opt.gpu == -1 then
      cuda = false
   else
      print(type(opt.gpu))
      cutorch.setDevice(opt.gpu)
      cuda = true
   end
   local model = torch.load(opt.modelDir.."/"..opt.modelName..".t7",
			    "binary")
   local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
   
   ----- define baseline scoring function -----------
   local cos = nn.CosineDistance()
   if cuda then
      cos:cuda()
      criterion:cuda()
   end
   function baseScorer(input, output)
      local emb = model.inLookup:forward(input)
      local inputEmb = torch.Tensor(emb:size()):copy(emb):cuda()
      emb = model.inLookup:forward(output)
      local outputEmb = torch.Tensor(emb:size()):copy(emb)
      if cuda then
	 inputEmb, outputEmb = inputEmb:cuda(), outputEmb:cuda()
      end
      inputEmb, outputEmb = inputEmb:sum(1):squeeze(1), outputEmb:sum(1):squeeze(1)
      local cosRes = cos:forward({inputEmb, outputEmb})
      local out = torch.Tensor(cosRes:size()):copy(cosRes)
      if cuda then
	 out = out:cuda()
      end
      return out
   end
   
   ------define model scorer--------------------- 
   -- local engine = tnt.SGDEngine()
   local baseMeter = tnt.ClassErrorMeter({topk = {1}, accuracy = true})
   local modMeter = tnt.ClassErrorMeter({topk = {1}, accuracy = true})

   local baselineAcc, modelAcc
   
   function getData(data)
      local input, target = data["input"], data["target"]
      local inp, outp, context = unpack(input)
      return inp, outp, context, target
   end

   function modelScorer(In, Out, Con, Target)
      local modProbs = model:forward({In, Out, Con})
      local scores, target = {}, {}
      for i=1,Target:size(2) do
	 local probs, targs = modProbs[{{}, i, {}}], Target[{{}, i}] 
	 local indZero = torch.find(targs, 0)
	 indZero = indZero[1] or (targs:size(1) + 1)
	 local probs, targs = probs[{{1, indZero -1}, {}}], targs[{{1, indZero -1}}]
	 scores[i] = -criterion:forward(probs, targs)
      end
      scores = torch.Tensor(scores)
      if cuda then
	 scores = scores:cuda()
      end
      return scores:reshape(scores:size(1),1)
   end

   local data = loadFromJSON(opt.evalFile, true)
   model:regenerateBeam(opt.beamSize, opt.maxLength)
   for i=1,#data do
      local example = data[i]
      local input = example["input"]
      local gate = example["context"]
      local resInput = torch.Tensor(input:size(1)):copy(input):resize(input:size(1), 1):cuda()
      local resGate = torch.Tensor(gate:size(1)):copy(gate):resize(1, gate:size(1)):cuda()
      local printOutput = model:decodeVocab(torch.totable(example["output"]), true)
      local printInput = model:decodeVocab(torch.totable(example["input"]), true)
      print("input:", resInput:type(), "gate:", resGate:type())
      local greedyModelOutput = model:greedy(resInput, resGate, 50)
      greedyModelOutput = model:decodeVocab(greedyModelOutput, true)
      local beamModelOutput = model:generate(resInput, resGate, opt.beamSize, opt.maxLen)
      beamModelOutput = model:decodeVocab(beamModelOutput, true)
      print("input:", printInput, "output:", printOutput,
	    "greedyModelOutput:", greedyModelOutput,
	    "beamModeloutput:", beamModelOutput)
   end
   local evalData = loadDataset(data, opt.batchSize, cuda, true)
   for sample in evalData() do
      print(count)
      local gold, noise = unpack(sample["input"])
      local goldIn, goldOut, goldCon, goldTarget = getData(gold)
      local noiseIn, noiseOut, noiseCon, noiseTarget = getData(noise)
      local goldScores = baseScorer(goldIn, goldOut)
      goldScores = goldScores:reshape(goldScores:size(1), 1)
      local noiseScores = {}
      for i=1,#noiseTarget do
	 local tempScores = baseScorer(noiseIn, noiseOut[i])
	 noiseScores[i] = tempScores:reshape(tempScores:size(1), 1)
      end
      local baseOut = torch.cat(goldScores, torch.cat(noiseScores))
      local _, baseMax = baseOut:max(2)
      print(baseMax:eq(1):sum())
      baseMeter:add(baseOut, sample["target"])
      
      local modGoldScores = modelScorer(goldIn, goldOut, goldCon, goldTarget)
      modGoldScores = modGoldScores:reshape(modGoldScores:size(1), 1)
      local modNoiseScores = {}
      for i=1,#noiseTarget do
	 print("noise: ", i)
	 local tempScores = modelScorer(noiseIn, noiseOut[i], noiseCon, noiseTarget[i])
	 modNoiseScores[i] = tempScores:reshape(tempScores:size(1), 1)
      end
      local modOut = torch.cat(modGoldScores, torch.cat(modNoiseScores))
      --print(modOut)
      local _, modMax = modOut:max(2)
      print(modMax:eq(1):sum())
      modMeter:add(modOut, sample["target"])
   end

   print("baseline acc.:", baseMeter:value(), "mod acc.:", modMeter:value())
end



local cmd = torch.CmdLine()

cmd:text()
cmd:text("NP negation evaluation:")
cmd:text()
------------------- Model Options ----------------
cmd:option("-beamSize", 5, "Beam size.")
cmd:option("-maxLen", 50, "Maximum sentence length for beam search.")
cmd:option("-modelDir", "/local/filespace/am2156/nncg/models", "Directory with model file.")
cmd:option("-modelName", "model", "Model filename.")
cmd:option("-gpu", -1, "GPU device to use. -1 for CPU.")
------------------ Data Options ------------------ 
cmd:option("-batchSize", 128, "Mini-batch size.")
cmd:option("-evalFile", "./data/mult_dev_data.json", "Model filename.")

local opt = cmd:parse(arg)
main(opt)
