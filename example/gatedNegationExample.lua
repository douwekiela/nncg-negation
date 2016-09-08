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
opt.jackknifeSize = 10
opt.numEpochs = 4
opt.batchSize = 8
opt.useGPU = true -- use CUDA_VISIBLE_DEVICES to set the GPU you want to use

-- define gated network graph
---- declare inputs
print("Constructing input layers")
local word_vector = nn.Identity()()
local gate_vector = nn.Identity()()

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
local loss = nn.MSECriterion()

-- GPU mode
if opt.useGPU then
    ged:cuda()
    loss:cuda()
end

-- read training data
print("Reading training data")
dofile("train.top10nns")

-- prepare data for 10-fold jackknife training
print ("Jackknifing data for training")
trainsize = table.getn(traindata)
shuffle_jackknife = torch.randperm(trainsize)
fold_size = math.modf(trainsize / opt.jackknifeSize)

local x, gradParameters = ged:getParameters()

for fold = 1,opt.jackknifeSize do
    -- reset model weights for this fold
    ged:reset()

    testdata_thisfold = {}
    traindata_thisfold = {}
    if fold == opt.jackknifeSize then
       max = trainsize
    else
       max = fold_size * fold
    end

    for i = 1, table.getn(traindata) do
    	if i >= (fold-1)*fold_size + 1 and i <= max then
	   table.insert(testdata_thisfold, traindata[shuffle_jackknife[i]])
	else
	   table.insert(traindata_thisfold, traindata[shuffle_jackknife[i]])
	end
    end

    -- train model
    for epoch = 1, opt.numEpochs do
    	print("Training fold", fold, "epoch", epoch)
	shuffle = torch.randperm(table.getn(traindata_thisfold))
	current_loss = 0
    	for t = 1, table.getn(traindata_thisfold), opt.batchSize do
    	    local inputs = {}
	    local gates = {}
	    local targets = {}
	    for j = t, math.min(t + opt.batchSize - 1, table.getn(traindata_thisfold)) do
	    	local input = traindata_thisfold[shuffle[j]][2]:resize(1, opt.embedSize)   -- start at idx 2 because the input word is in the table
	    	local gate = traindata_thisfold[shuffle[j]][3]:resize(1, opt.embedSize)
	    	local target = traindata_thisfold[shuffle[j]][4]:resize(1, opt.embedSize)

            if opt.useGPU then
                input = input:cuda()
                gate = gate:cuda()
                target = target:cuda()
            end

		table.insert(inputs, input)
		table.insert(gates, gate)
		table.insert(targets, target)
            end

	    local feval = function(w)  -- w = weight vector. returns loss, dloss_dw
	        gradParameters:zero()
	        local f = 0  -- for averaging error
	        for k = 1, #inputs do
	      	  local output = ged:forward({inputs[k], gates[k]})
		  local err = loss:forward(output, targets[k])
		  f = f + err
		  local gradErr = loss:backward(output, targets[k])
		  ged:backward({inputs[k], gates[k]}, gradErr)
	        end -- for k = 1, #inputs

	        -- normalize gradients and f(X)
	        gradParameters:div(#inputs)
	        f = f/(#inputs)
	        return f, gradParameters
	    end -- local feval

	    _, fs = optim.adadelta(feval, x, {rho = 0.9})
	    current_loss = current_loss + fs[1]

	end -- for t = 1, table.getn(traindata_thisfold), opt.batchSize

	current_loss = current_loss / table.getn(traindata_thisfold)
    	print("... Current loss", current_loss)

    end   -- for epoch = 1, opt.numEpochs

    -- predict for this fold
    predictions = {}
    for t = 1, table.getn(testdata_thisfold) do
    	local input_word = testdata_thisfold[t][1]
        local output = ged:forward({testdata_thisfold[t][2]:resize(1, opt.embedSize), testdata_thisfold[t][3]:resize(1, opt.embedSize)})
	local target = testdata_thisfold[t][4]:resize(1, opt.embedSize)
	table.insert(predictions, {input_word, output, target})
    end
    print("Saving model and predictions")
    torch.save("predict." .. fold, predictions)
    torch.save("model." .. fold .. ".net", ged)

end   -- for fold = 1, opt.jackknifeSize


-- uncomment below for the original set of dummy test vectors
-- -- test example
-- ---- construct dummy vector
-- cold_vector = torch.rand(1, opt.embedSize)
-- hot_vector = torch.rand(1, opt.embedSize)
-- cold_gate = torch.rand(1, opt.gateSize)
-- 
-- ---- predict antonym vector
-- local predict_antonym = ged:forward{cold_vector, cold_gate}
-- ---- compute loss
-- local error = loss:forward(predict_antonym, hot_vector)
-- 
-- ---- print loss
-- print("Error: ", error)

