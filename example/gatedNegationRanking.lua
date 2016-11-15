local cutorch = require "cutorch"
local cunn = require "cunn"
local nn = require "nn"
local nngraph = require "nngraph"
local optim = require "optim"
local js = require "json"
local tnt = require "torchnet"
local moses = require "moses"

-- set gated network hyperparameters
local opt = {}
opt.embedSize = 300
opt.gateSize = 300
opt.hiddenSize = 600
opt.jackknifeSize = 10
opt.numEpochs = 5000
opt.batchSize = 16
opt.useGPU = true -- use CUDA_VISIBLE_DEVICES to set the GPU you want to use
opt.predFile = "predict.out"
opt.trainFile = "data/pairs.txt"
opt.mcqFile = "data/mcq_dev.txt"
opt.embedFile = "data/train_combined.json"
opt.gateFile = "data/train_combined_gates.json"
opt.evalFreq = 10000

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
local loss = nn.MarginRankingCriterion(0.5)({parInput({cos1, cos2}), rank})
local loss_module = nn.gModule({out_vec, targ_vec, sample_vec, rank}, {loss})

-- GPU mode
if opt.useGPU then
    ged:cuda()
    loss_module:cuda()
end

-- read training and test data
print("Reading training data")
-- traindata = torch.Tensor(19578, 4*opt.embedSize)
traindata = torch.Tensor(39154, 4*opt.embedSize)
io.input('data.top10nns.raw.ranking.negsample.train')
linecount = 0
for line in io.lines() do
    linecount = linecount + 1
    valcount = 0
    for num in string.gmatch(line, "%S+") do
    	valcount = valcount + 1
	if valcount > 1 then
       	   traindata[linecount][valcount - 1] = tonumber(num)
	end
    end
end    
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
	 local index = function(k, tab) return tab[k] end
	 local mergecl = function(source, closure)
	    return tnt.utils.table.mergetensor(
		  tnt.utils.table.foreach{
		     tbl = batch[source],
		     closure = closure
	    })
	 end
	 return {
	    input = {
	       mergecl("input", moses.bind(index, 1)),
	       mergecl("input", moses.bind(index, 2))
	    },
	    target = {
	       mergecl("target", moses.bind(index, 1)),
	       mergecl("target", moses.bind(index, 2))
	    }
	 }
      end
   }
   return tnt.DatasetIterator{dataset = outData}
end
print("Reading test data")
testdata = torch.Tensor(225, 4*opt.embedSize)
testwords = {}
io.input('data.top10nns.raw.ranking.negsample.test')
linecount = 0
for line in io.lines() do
    linecount = linecount + 1
    valcount = 0
    for num in string.gmatch(line, "%S+") do
    	valcount = valcount + 1
	if valcount == 1 then
	   testwords[linecount] = num
	else
       	   testdata[linecount][valcount - 1] = tonumber(num)
	end
    end
end

function loadMCQData(filePath, batchSize)
   local outData = tnt.ListDataset{
      filename = filePath,
      load = function(line)
	 local word, candidate1, candidate2, candidate3, candidate4, candidate5, antonym =
	    line:match("(%S+): (%S+) (%S+) (%S+) (%S+) (%S+) :: (%S+)")
	 local res = moses.indexOf({candidate1, candidate2, candidate3, candidate4, candidate5}, antonym)
	 end
	 return {
	    input = {{embDict[word],
		      gateDict[word]},
	       embDict[candidate1],
	       embDict[candidate2],
	       embDict[candidate3],
	       embDict[candidate4],
	       embDict[candidate5]
	       },
	    target = res
	 }
      end
   }
   outData = tnt.BatchDataset{
      batchsize = batchSize,
      dataset = outData,
      merge = function(batch)
	 local index = function(k, tab) return tab[k] end
	 local mergecl = function(source, closure)
	    return tnt.utils.table.mergetensor(
		  tnt.utils.table.foreach{
		     tbl = batch[source],
		     closure = closure
	    })
	 end
	 return {
	    input = {
	       {tnt.utils.table.mergetensor(
		  tnt.utils.table.foreach{
		     tbl =tnt.utils.table.foreach{
			tbl = batch["input"],
			closure = moses.bind(index, 1)},
		     closure = moses.bind(index, 1)}),
	       tnt.utils.table.mergetensor(
		  tnt.utils.table.foreach{
		     tbl = tnt.utils.table.foreach{
			tbl = batch["input"],
			closure = moses.bind(index, 1)},
		     closure = moses.bind(index, 2)
	       })},
	       mergecl("input", moses.bind(index, 2)),
	       mergecl("input", moses.bind(index, 3)),
	       mergecl("input", moses.bind(index, 4)),
	       mergecl("input", moses.bind(index, 5)),
	       mergecl("input", moses.bind(index, 6))
	    },
	    target = torch.Tensor(batch["target"])
	 }
      end
   }
   return tnt.DatasetIterator{dataset = outData}
end

print("Loading MCQ data...")
local mcqData = loadMCQData(opt.mcqFile, opt.batchSize)
local accMeter = tnt.ClassErrorMeter{topk={1}, accuracy = true}

function eval()
   accMeter:reset()
   local cos = nn.CosineDistance()
   if opt.useGPU then cos:cuda() end
   for example in mcqData() do
      local input, candidate1, candidate2, candidate3, candidate4, candidate5 = unpack(example["input"])
      local inputs, gates = unpack(input)
      local result = ged:forward({inputs, gates})
      local score = function(vec)
      	 return cos:forward{result, vec} 
      end
      local scores = tnt.utils.table.mergetensor(
      	 tnt.utils.table.foreach{
      	    tbl = {candidate1, candidate2, candidate3, candidate4, candidate5},
      	    closure = score    
      }):transpose(1, 2)
      accMeter:add(scores, example["target"])
   end
   return accMeter:value()
end

local accs = {}
-- train model
local x, gradParameters = ged:getParameters()

for epoch = 1, opt.numEpochs do
    print("Training epoch", epoch)
    shuffle = torch.randperm(traindata:size()[1])
    current_loss = 0
    for t = 1, traindata:size()[1], opt.batchSize do
--        print("example number: ", t)
        local inputs = {}
	local gates = {}
	local targets = {}
	local samples = {}
	for j = t, math.min(t + opt.batchSize - 1, traindata:size()[1]) do
	    local input = traindata[shuffle[j]]:narrow(1, 1, opt.embedSize):resize(1, opt.embedSize)
	    local gate = traindata[shuffle[j]]:narrow(1, opt.embedSize + 1, opt.embedSize):resize(1, opt.embedSize)
	    local target = traindata[shuffle[j]]:narrow(1, 2*opt.embedSize + 1, opt.embedSize):resize(1, opt.embedSize)
	    local sample = traindata[shuffle[j]]:narrow(1, 3*opt.embedSize + 1, opt.embedSize):resize(1,opt.embedSize)

	    if opt.useGPU then
                input = input:cuda()
                gate = gate:cuda()
                target = target:cuda()
                sample = sample:cuda()
            end

	    table.insert(inputs, input:clone())
	    table.insert(gates, gate:clone())
	    table.insert(targets, target:clone())
	    table.insert(samples, sample:clone())
        end

	local feval = function(w)  -- w = weight vector. returns loss, dloss_dw
	      gradParameters:zero()
	      inputs = torch.cat(inputs, 1)
	      gates = torch.cat(gates, 1)
	      targets = torch.cat(targets, 1)
	      samples = torch.cat(samples, 1)
	      ranks = torch.ones(inputs:size(1)):cuda()
	      
	      local result = ged:forward({inputs, gates})
	      local f = loss_module:forward({result, targets, samples, ranks})
	      local gradErr =  loss_module:backward({result, targets, samples, ranks},
		 torch.ones(1):cuda())
	      local gradOut, gradTarg, gradSample, gradRank = unpack(gradErr)
	      ged:backward({inputs, gates}, gradOut)

	      -- DON'T normalize gradients and f(X)
	      -- gradParameters:div(inputs:size(1))
	      return f, gradParameters
	end -- local feval

	_, fs = optim.adadelta(feval, x, {rho = 0.9})
	current_loss = current_loss + fs[1]

       _, fs = optim.adadelta(feval, x, {rho = 0.9})
       current_loss = current_loss + fs[1]
       if (t*opt.batchSize) % opt.evalFreq == 0 then
	  acc = eval()[1]
	  print("Accuracy: ", acc)
	  table.insert(accs, {epoch = epoch, t = t*opt.batchSize, acc=acc})
       end
       t = t + 1
       print("... Avg. loss", current_loss/t)
       
    end -- for t = 1, traindata:size()[1], opt.batchSize

    current_loss = (current_loss * opt.batchSize) / traindata:size()[1]
    print("... Current loss", current_loss[1])

    if epoch % 50 == 0 then
       print("Saving model")
       torch.save("model_marg05_batch16." .. tostring(epoch) .. ".net", ged)
    end

end   -- for epoch = 1, opt.numEpochs

-- predict
print "Predicting"
predFileStream = io.open(opt.predFile, "w") 
-- module with the first half of the network
local shuffle = torch.randperm(testdata:size(1))
for t = 1, testdata:size(1) do
    local input_word = testwords[t]
    local input = testdata[shuffle[t]]:narrow(1, 1, opt.embedSize):resize(1, opt.embedSize)
    local gate = testdata[shuffle[t]]:narrow(1, opt.embedSize + 1, opt.embedSize):resize(1, opt.embedSize)
    if opt.useGPU then
       input = input:cuda()
       gate = gate:cuda()
    end
    local output = ged:forward({input, gate})
    predFileStream:write(input_word .. "\t[")
    for k = 1, output:size(2) do
       predFileStream:write(output[1][k] .. ", ")
    end
    predFileStream:write("]\n")
end

