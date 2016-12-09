local cutorch = require "cutorch"
local cunn = require "cunn"
local nn = require "nn"
local nngraph = require "nngraph"
local argparse = require "argparse"

-- get command line arguments
local parser = argparse()
parser:flag("--gates", "whether to concatenate the gate for input to the model")
parser:option("--test", "which test file to use")
parser:option("--model", "which model to use")
local args = parser:parse()

-- set parameters
local opt = {}
opt.testFile = '/local/filespace/lr346/disco/experiments/negation/nncg-negation/traindata/' .. args['test'] .. '.test'
opt.modelFile = '/local/filespace/lr346/disco/experiments/negation/nncg-negation/eacl17/models/' .. args['test'] .. '/model_' .. args['model'] .. '.net'
opt.predFile = '/local/filespace/lr346/disco/experiments/negation/nncg-negation/eacl17/predictions/' .. args['test'] .. '/model_' .. args['model'] .. '.out'
opt.forwardGates = args['gates']
opt.embedSize = 300
opt.gateSize = 300
opt.useGPU = true -- use CUDA_VISIBLE_DEVICES to set the GPU you want to use

-- read model
print ("Reading model", opt.modelFile)
ged = torch.load(opt.modelFile)

print("Counting test examples")
io.input(opt.testFile)
linecount = 0
for line in io.lines() do
    linecount = linecount + 1
end
io.input():close()
opt.numTestExx = linecount
print("...", tostring(opt.numTestExx))

print ("Reading test data")
testdata = torch.Tensor(opt.numTestExx, 2*opt.embedSize)
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

print "Predicting"
print(ged)
predFileStream = io.open(opt.predFile, "w") 
for t = 1, testdata:size(1) do
    local input_word = testwords[t]
    local input = testdata[t]:narrow(1, 1, opt.embedSize):resize(1, opt.embedSize)
    local gate = testdata[t]:narrow(1, opt.embedSize + 1, opt.embedSize):resize(1, opt.embedSize)
    if opt.useGPU then
       input = input:cuda()
       gate = gate:cuda()
    end
    local output = torch.zeros(opt.embedSize)
    if opt.forwardGates then
        output = ged:forward(torch.cat(input, gate))
    else
	output = ged:forward(input)
    end
    predFileStream:write(input_word .. "\t[")
    for k = 1, output:size(2) do
       predFileStream:write(output[1][k] .. ", ")
    end
    predFileStream:write("]\n")
end

