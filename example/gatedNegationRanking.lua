local cutorch = require "cutorch"
local cunn = require "cunn"
local nn = require "nn"
local nngraph = require "nngraph"
local optim = require "optim"
local js = require "json"
local tnt = require "torchnet"

-- set gated network hyperparameters
local opt = {}
opt.embedSize = 300
opt.gateSize = 300
opt.hiddenSize = 600
opt.jackknifeSize = 10
opt.numEpochs = 1
opt.batchSize = 100
opt.useGPU = true -- use CUDA_VISIBLE_DEVICES to set the GPU you want to use
opt.predFile = "predict.out"
opt.trainFile = "data/pairs.txt"
opt.embedFile = "data/train_combined.json"
opt.gateFile = "data/train_combined_gates.json"

-- define gated network graph
---- declare inputs
print("Constructing input layers")
local word_vector = nn.Identity()()
local gate_vector = nn.Identity()()
local targ_vector = nn.Identity()()
local sample_vector = nn.Identity()()

---- define hidden layer
print("Constructing hidden state")
local h = nn.Sigmoid()(nn.Bilinear(opt.embedSize, opt.gateSize, opt.hiddenSize)({word_vector, gate_vector}))

---- define output layer
print("Constructing output")
local output = nn.Bilinear(opt.hiddenSize, opt.gateSize, opt.embedSize)({h, gate_vector})

---- Construct model
print("Constructing module")
local ged = nn.gModule({word_vector, gate_vector}, {output})

-- define loss function
print("Defining loss function")
local out_vec = nn.Identity()()
local targ_vec = nn.Identity()()
local sample_vec = nn.Identity()()
local rank = nn.Identity()()
local cos1 = nn.CosineDistance()({out_vec, targ_vec})
local cos2 = nn.CosineDistance()({out_vec, sample_vec})
local parInput = nn.ParallelTable()
   :add(nn.Identity())
   :add(nn.Identity())
local loss = nn.MarginRankingCriterion()({parInput({cos1, cos2}), rank})
local loss_module = nn.gModule({out_vec, targ_vec, sample_vec, rank}, {loss})

-- GPU mode
if opt.useGPU then
    ged:cuda()
    loss_module:cuda()
end

-- read training and test data
-- function to load embedding dictionary
function loadEmbDict(filePath, gpu)
   local data = js.decode(io.open(filePath, "r"):read("*all"))
   local keys = {}
   for word, vector in pairs(data) do
      table.insert(keys, word)
      data[word] = torch.Tensor(vector)
      if gpu then
	 data[word] = data[word]:cuda()
      end
   end
   return data, keys 
end

-- load embeddings and gates
print("Loading word embeddings...")
local embDict, embKeys = loadEmbDict(opt.embedFile, opt.useGPU)
print("Loading gates...")
local gateDict, _ = loadEmbDict(opt.gateFile, opt.useGPU)

-- function to load pairs data as an iterator
function loadPairData(filePath, batchSize)
   local gen = torch.Generator()
   local outData = tnt.ListDataset{
      filename = filePath,
      load = function(line)
	 local word, antonym = line:match("(%S+) (%S+)") 
	 return {
	    input = {embDict[word],
		     gateDict[word]},
	    target = {embDict[antonym],
		      embDict[embKeys[torch.random(gen, 1, #embKeys)]]}
	 }
      end
   }
   outData = tnt.BatchDataset{
      batchsize = batchSize,
      dataset = outData,
      merge = function(batch)
	 local first = function(tab) return tab[1] end
	 local second = function(tab) return tab[2] end
	 local mergecl = function(source, closure)
	    return tnt.utils.table.mergetensor(
		  tnt.utils.table.foreach{
		     tbl = batch[source],
		     closure = closure
	    })
	 end
	 return {
	    input = {
	       mergecl("input", first),
	       mergecl("input", second)
	    },
	    target = {
	       mergecl("target", first),
	       mergecl("target", second)
	    }
	 }
      end
   }
   return tnt.DatasetIterator{dataset = outData}
end

-- load pairs training data
print("Loading training data...")
trainPairs = loadPairData(opt.trainFile, opt.batchSize)

-- train model
local x, gradParameters = ged:getParameters()

for epoch = 1, opt.numEpochs do
    print("Training epoch", epoch)
    current_loss = 0
    local t = 0
    for example in trainPairs() do
       local inputs, gates = unpack(example["input"])
       local targets, samples = unpack(example["target"])
       local feval = function(w)  -- w = weight vector. returns loss, dloss_dw
	  gradParameters:zero()
	  ranks = torch.ones(inputs:size(1)):cuda()
	      
	  local result = ged:forward({inputs, gates})
	  local f = loss_module:forward({result, targets, samples, ranks})
	  local gradErr =  loss_module:backward({result, targets, samples, ranks},
	     torch.ones(1):cuda())
	  local gradOut, gradTarg, gradSample, gradRank = unpack(gradErr)
	  ged:backward({inputs, gates}, gradOut)
	  
	  -- normalize gradients and f(X)
	  gradParameters:div(inputs:size(1))
	  return f, gradParameters
       end -- local feval

       _, fs = optim.adadelta(feval, x, {rho = 0.9})
       current_loss = current_loss + fs[1]
       t = t + 1
       print("... Avg. loss", current_loss/t)
       
    end -- for t = 1, traindata:size()[1], opt.batchSize

    current_loss = current_loss / t 
    print("... Current loss", current_loss)

end   -- for epoch = 1, opt.numEpochs

-- -- predict
-- print "Predicting"
-- predFileStream = io.open(opt.predFile, "w") 
-- -- module with the first half of the network
-- local shuffle = torch.randperm(testdata:size(1))
-- for t = 1, testdata:size(1) do
--     local input_word = testwords[t]
--     local input = testdata[shuffle[t]]:narrow(1, 1, opt.embedSize):resize(1, opt.embedSize)
--     local gate = testdata[shuffle[t]]:narrow(1, opt.embedSize + 1, opt.embedSize):resize(1, opt.embedSize)
--     if opt.useGPU then
--        input = input:cuda()
--        gate = gate:cuda()
--     end
--     local output = ged:forward({input, gate}):squeeze()
--     print("output", output:size())
--     predFileStream:write(input_word .. "\t[")
--     for k = 1, output:size(1) do
--        predFileStream:write(output[k] .. ", ")
--     end
--     predFileStream:write("]\n")
-- end
print("Saving model")
torch.save("model.net", ged)

