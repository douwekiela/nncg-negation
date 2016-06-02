local nn = require "nn"
local nngraph = require "nngraph"

-- set gated network hyperparameters
local opt = {}
opt.embedSize = 10
opt.gateSize = 15
opt.hiddenSize = 5

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

-- test example
---- construct dummy vector
cold_vector = torch.rand(1, opt.embedSize)
hot_vector = torch.rand(1, opt.embedSize)
cold_gate = torch.rand(1, opt.gateSize)

---- predict antonym vector
local predict_antonym = ged:forward{cold_vector, cold_gate}
---- compute loss
local error = loss:forward(predict_antonym, hot_vector)

---- print loss
print("Error: ", error)

