local cutorch = require "cutorch"
local cunn = require "cunn"
local nn = require "nn"
local optim = require "optim"
local argparse = require "argparse"

-- get command line arguments
local parser = argparse()
parser:flag("--dropout", "whether to use dropout")
parser:option("--learning-rate", "learning rate", 0.99)
  :convert(tonumber)				 	   
parser:option("--hidden-size", "hidden layer size", 300)
  :convert(tonumber)				 	   
parser:option("--batch-size", "batch size", 48)
  :convert(tonumber)				 	   
parser:option("--epochs", "number of epochs", 500)
  :convert(tonumber)				 	   
parser:option("--train", "which training file to use")
parser:flag("--gates", "whether to concatenate the gate with the input vector")
parser:flag("--logging")
local args = parser:parse()

-- set hyperparameters
local opt = {}
opt.gates = args['gates']
opt.dropout = args['dropout'] 
opt.embedSize = 300
opt.gateSize = 300
opt.hiddenSize = args['hidden_size']
opt.batchSize = args['batch_size']
opt.numEpochs = args['epochs']
opt.learningRate = args['learning_rate']
opt.saveEpochInterval = 100
opt.useGPU = true -- use CUDA_VISIBLE_DEVICES to set the GPU you want to use
opt.trainFile = '/local/filespace/lr346/disco/experiments/negation/nncg-negation/traindata/' .. args['train'] .. '.train'
opt.modelBase = '/local/filespace/lr346/disco/experiments/negation/nncg-negation/eacl17/models/' .. args['train'] .. '/model_ff_mse_' .. tostring(opt.hiddenSize) .. '_' .. tostring(opt.learningRate) .. '_' .. tostring(opt.batchSize)
opt.logFile = '/local/filespace/lr346/disco/experiments/negation/nncg-negation/eacl17/models/' .. args['train'] .. '/log_ff_mse_' .. tostring(opt.hiddenSize) .. '_' .. tostring(opt.learningRate) .. '_' .. tostring(opt.batchSize)
if opt.gates then
  opt.modelBase = opt.modelBase .. '_withgates'
  opt.logFile = opt.logFile .. '_withgates'
else
  opt.modelBase = opt.modelBase .. '_nogates'
  opt.logFile = opt.logFile .. '_nogates'
end
if opt.dropout then
  opt.modelBase = opt.modelBase .. '_drop'
else
  opt.modelBase = opt.modelBase .. '_nodrop'
end
if args['logging'] then
   logfile = io.open(opt.logFile, 'w')
end

print("Counting training examples")
io.input(opt.trainFile)
linecount = 0
for line in io.lines() do
    linecount = linecount + 1
end
io.input():close()
opt.numTrainingExx = linecount
print("...", tostring(opt.numTrainingExx))

print("Constructing network")
inputSize = opt.embedSize
if opt.gates then
   inputSize = opt.embedSize + opt.gateSize
end
local ged = nn.Sequential()
ged:add(nn.Linear(inputSize, opt.hiddenSize))
ged:add(nn.Sigmoid())
ged:add(nn.Linear(opt.hiddenSize, opt.embedSize))

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
traindata = torch.Tensor(opt.numTrainingExx, 3*opt.embedSize)
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

-- train model
local x, gradParameters = ged:getParameters()

for epoch = 1, opt.numEpochs do
    print("Training epoch", epoch)
    if args['logging'] then
       logfile:write("Training epoch ", epoch, "\n")
    end
    shuffle = torch.randperm(traindata:size(1))
    local current_loss = 0
    for t = 1, traindata:size(1), opt.batchSize do
        local inputs = {}
	local targets = {}
	local samples = {}
	for j = t, math.min(t + opt.batchSize - 1, traindata:size()[1]) do
	    local input = traindata[shuffle[j]]:narrow(1, 1, opt.embedSize):resize(1, opt.embedSize)
	    local gate = traindata[shuffle[j]]:narrow(1, opt.embedSize + 1, opt.embedSize):resize(1, opt.embedSize)
	    local target = traindata[shuffle[j]]:narrow(1, 2*opt.embedSize + 1, opt.embedSize):resize(1, opt.embedSize)
	    if opt.useGPU then
                input = input:cuda()
                gate = gate:cuda()
                target = target:cuda()
            end

	    if opt.gates then
                table.insert(inputs, torch.cat(input:clone(), gate:clone()))
            else
                table.insert(inputs, input:clone())
	    end
	    table.insert(targets, target:clone())
        end

	local feval = function(w)  -- w = weight vector. returns loss, dloss_dw
	      gradParameters:zero()
	      inputs = torch.cat(inputs, 1)
	      targets = torch.cat(targets, 1)
	      
	      local result = ged:forward(inputs)
	      local f = loss:forward(result, targets)
	      local gradErr = loss:backward(result, targets)
	      ged:backward(inputs, gradErr)

	      return f, gradParameters
	end -- local feval

	_, fs = optim.adadelta(feval, x, {rho = opt.learningRate})
	current_loss = current_loss + fs[1]

    end -- for t = 1, traindata:size()[1], opt.batchSize

    current_loss = (current_loss * opt.batchSize) / traindata:size(1)
    print("... Current loss", current_loss)
    if args['logging'] then
       logfile:write("Current loss ", current_loss, "\n")
    end

    if epoch % opt.saveEpochInterval == 0 then
       print("Saving model")
       torch.save(opt.modelBase .. "_epoch" .. tostring(epoch) .. ".net", ged)
    end

end   -- for epoch = 1, opt.numEpochs


