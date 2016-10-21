local cutorch = require "cutorch"
local cunn = require "cunn"
local nn = require "nn"
local nngraph = require "nngraph"

local opt = {}
opt.modelFile = arg[1]
opt.predFile = arg[1] .. ".out"

-- set gated network hyperparameters
opt.embedSize = 300
opt.gateSize = 300
opt.useGPU = true -- use CUDA_VISIBLE_DEVICES to set the GPU you want to use

-- read model
print ("Reading model", opt.modelFile)
ged = torch.load(opt.modelFile)

-- read test data
print ("Reading test data")
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

-- predict
print "Predicting"
predFileStream = io.open(opt.predFile, "w") 
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

