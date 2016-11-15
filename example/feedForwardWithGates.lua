local cutorch = require "cutorch"
local cunn = require "cunn"
local nn = require "nn"
local nngraph = require "nngraph"
local optim = require "optim"

-- set gated network hyperparameters
local opt = {}
opt.embedSize = 300
opt.gateSize = 300
opt.hiddenSize = 600
opt.batchSize = 100
opt.numEpochs = 10
opt.saveEpochInterval = 1
opt.margin = 0.5
opt.useGPU = true -- use CUDA_VISIBLE_DEVICES to set the GPU you want to use
opt.predFile = "predict.mtrain_withgates.out"
opt.numTrainingExx = 198302
opt.numTestExx = 186
opt.trainFile = '/local/filespace/lr346/disco/experiments/negation/nncg-negation/mohammad/mtrain.train'
opt.testFile = '/local/filespace/lr346/disco/experiments/negation/nncg-negation/mohammad/mtrain.test'
opt.modelBase = 'model_mtrain'

-- define gated network graph
---- declare inputs
print("Constructing input layers")
local word_vector = nn.Identity()()
local gate_vector = nn.Identity()()
local wn_targ_vector = nn.Identity()()
local m_targ_vector = nn.Identity()()
local sample_vector = nn.Identity()()

--local concat_input_gate = nn.Concat()({word_vector, gate_vector})

---- define hidden layer
print("Constructing hidden state")
local h = nn.Sigmoid()(nn.Linear(opt.embedSize+opt.gateSize, opt.hiddenSize)(nn.JoinTable(2)({word_vector, gate_vector})))

---- define output layer
print("Constructing output")
local output = nn.Linear(opt.hiddenSize, opt.embedSize)(nn.Sigmoid()({h}))

---- Construct model
print("Constructing module")
local ged = nn.gModule({word_vector, gate_vector}, {output})

-- define loss function
print("Defining loss function")
local out_vec = nn.Identity()()
local out_vec2 = nn.Identity()()
local wn_targ_vec = nn.Identity()()
local m_targ_vec = nn.Identity()()
local sample_vec = nn.Identity()()
local rank = nn.Identity()()
local cos1 = nn.CosineDistance()({out_vec, m_targ_vec})
local cos2 = nn.CosineDistance()({out_vec, sample_vec})

local parInput = nn.ParallelTable()
   :add(nn.Identity())
   :add(nn.Identity())
local ranking_loss = nn.MarginRankingCriterion(opt.margin)({parInput({cos1, cos2}), rank})

local split1 = nn.SplitTable(1)
local split2 = nn.SplitTable(1)

local mse_loss = nn.MSECriterion()({out_vec2, wn_targ_vec})

-- local loss = nn.ParallelCriterion()({mse_loss, ranking_loss})
-- local loss_module = nn.gModule({out_vec, wn_targ_vec, m_targ_vec, sample_vec, rank}, {loss})

local mse_loss_module = nn.gModule({out_vec2, wn_targ_vec}, {mse_loss})

local ranking_loss_module = nn.gModule({out_vec, m_targ_vec, sample_vec, rank}, {ranking_loss})

-- GPU mode
if opt.useGPU then
    ged:cuda()
--    loss_module:cuda()
    mse_loss_module:cuda()
    ranking_loss_module:cuda()
end

-- read training and test data
print("Reading training data")
traindata = torch.Tensor(opt.numTrainingExx, 5*opt.embedSize)
io.input(opt.trainFile)
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
print("Reading test data")
testdata = torch.Tensor(opt.numTestExx, 5*opt.embedSize)
testwords = {}
io.input(opt.testFile)
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
	local wn_targets = {}
	local m_targets = {}
	local samples = {}
	for j = t, math.min(t + opt.batchSize - 1, traindata:size()[1]) do
	    local input = traindata[shuffle[j]]:narrow(1, 1, opt.embedSize):resize(1, opt.embedSize)
	    local gate = traindata[shuffle[j]]:narrow(1, opt.embedSize + 1, opt.embedSize):resize(1, opt.embedSize)
	    local wn_target = traindata[shuffle[j]]:narrow(1, 2*opt.embedSize + 1, opt.embedSize):resize(1, opt.embedSize)
	    local m_target = traindata[shuffle[j]]:narrow(1, 3*opt.embedSize + 1, opt.embedSize):resize(1, opt.embedSize)
	    local sample = traindata[shuffle[j]]:narrow(1, 4*opt.embedSize + 1, opt.embedSize):resize(1,opt.embedSize)

	    if opt.useGPU then
                input = input:cuda()
                gate = gate:cuda()
                wn_target = wn_target:cuda()
                m_target = m_target:cuda()
                sample = sample:cuda()
            end

	    table.insert(inputs, input:clone())
	    table.insert(gates, gate:clone())
	    table.insert(wn_targets, wn_target:clone())
	    table.insert(m_targets, m_target:clone())
	    table.insert(samples, sample:clone())
        end

	local feval = function(w)  -- w = weight vector. returns loss, dloss_dw
	      gradParameters:zero()
	      inputs = torch.cat(inputs, 1)
	      gates = torch.cat(gates, 1)
	      wn_targets = torch.cat(wn_targets, 1)
	      m_targets = torch.cat(m_targets, 1)
	      samples = torch.cat(samples, 1)
	      ranks = torch.ones(inputs:size(1)):cuda()
	      
--	      local result = ged:forward({inputs, gates})
--	      local f = loss_module:forward({result, wn_targets, m_targets, samples, ranks})
--	      local gradErr =  loss_module:backward({result, wn_targets, m_targets, samples, ranks}, torch.ones(1):cuda())
--	      local gradOut, gradWnTarg, gradMTarg, gradSample, gradRank = unpack(gradErr)
--	      ged:backward({inputs, gates}, gradOut)

	      local result = ged:forward({inputs, gates})
	      local mse_f = mse_loss_module:forward({result, wn_targets})
--	      print(mse_f)
	      local ranking_f = ranking_loss_module:forward({result, m_targets, samples, ranks})
--	      print(ranking_f)
	      local ranking_gradErr =  ranking_loss_module:backward({result, m_targets, samples, ranks}, torch.ones(1):cuda())
	      local mse_gradErr =  mse_loss_module:backward({result, wn_targets}, torch.ones(1):cuda())
	      local mse_gradOut, mse_gradWnTarg = unpack(mse_gradErr)
	      local ranking_gradOut, ranking_gradMTarg, ranking_gradSample, ranking_gradRank = unpack(ranking_gradErr)
	      ged:backward({inputs, gates}, (mse_gradOut+ranking_gradOut)/2)

	      -- DON'T normalize gradients and f(X)
	      -- gradParameters:div(inputs:size(1))
	      return (ranking_f+mse_f)/2, gradParameters
	end -- local feval

	_, fs = optim.adadelta(feval, x, {rho = 0.9})
	current_loss = current_loss + fs[1]

    end -- for t = 1, traindata:size()[1], opt.batchSize

    current_loss = (current_loss * opt.batchSize) / traindata:size()[1]
    print("... Current loss", current_loss[1])

    if epoch % opt.saveEpochInterval == 0 then
       print("Saving model")
       torch.save(opt.modelBase .. tostring(epoch) .. ".net", ged)
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

