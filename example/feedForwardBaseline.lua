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
opt.numEpochs = 100
opt.batchSize = 8
opt.useGPU = true -- use CUDA_VISIBLE_DEVICES to set the GPU you want to use

-- define gated network graph
---- declare inputs
print("Constructing input layers")
local word_vector = nn.Identity()()

---- define hidden layer
print("Constructing hidden state")
local h = nn.Sigmoid()(nn.Linear(opt.embedSize, opt.hiddenSize)(word_vector))

---- define output layer
print("Constructing output")
local output = nn.Linear(opt.hiddenSize, opt.embedSize)(h)

---- Construct model
print("Constructing module")
local ged = nn.gModule({word_vector}, {output})

-- define loss function
print("Defining loss function")
local loss = nn.MSECriterion()

-- GPU mode
if opt.useGPU then
    ged:cuda()
    loss:cuda()
end

-- read training and test data
print("Reading training data")
dofile("data.train.top10nns.rand10pct")
print("Reading test data")
dofile("data.test.top10nns.rand10pct")

-- train model
local x, gradParameters = ged:getParameters()

for epoch = 1, opt.numEpochs do
    print("Training epoch", epoch)
    shuffle = torch.randperm(table.getn(traindata))
    current_loss = 0
    for t = 1, table.getn(traindata), opt.batchSize do
        local inputs = {}
	local targets = {}
	for j = t, math.min(t + opt.batchSize - 1, table.getn(traindata)) do
	    local input = traindata[shuffle[j]][2]:resize(1, opt.embedSize)   -- start at idx 2 because the input word is in the table
	    local target = traindata[shuffle[j]][4]:resize(1, opt.embedSize)

	    if opt.useGPU then
                input = input:cuda()
                target = target:cuda()
            end

	    table.insert(inputs, input:clone())
	    table.insert(targets, target:clone())
        end

	local feval = function(w)  -- w = weight vector. returns loss, dloss_dw
	      gradParameters:zero()
	      local f = 0  -- for averaging error
	      for k = 1, #inputs do
	      	  local output = ged:forward(inputs[k])
		  local err = loss:forward(output, targets[k])
		  f = f + err
		  local gradErr = loss:backward(output, targets[k])
		  ged:backward(inputs[k], gradErr)
	      end -- for k = 1, #inputs

	      -- normalize gradients and f(X)
	      gradParameters:div(#inputs)
	      f = f/(#inputs)
	      return f, gradParameters
	end -- local feval

	_, fs = optim.adadelta(feval, x, {rho = 0.9})
	current_loss = current_loss + fs[1]

    end -- for t = 1, table.getn(traindata), opt.batchSize

    current_loss = current_loss / table.getn(traindata)
    print("... Current loss", current_loss)

end   -- for epoch = 1, opt.numEpochs

-- predict
print "Predicting"
predictions = {}
for t = 1, table.getn(testdata) do
    local input_word = testdata[t][1]
    local input = testdata[t][2]:resize(1, opt.embedSize)
    if opt.useGPU then
       input = input:cuda()
    end
    local output = ged:forward(input)
    table.insert(predictions, {input_word, output:clone()})
end
print("Saving model and predictions")
torch.save("ffBase.predict.out", predictions)
torch.save("ffBase.model.net", ged)







