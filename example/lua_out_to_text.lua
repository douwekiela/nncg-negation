require "cutorch"
require "cunn"

pred = torch.load("predict.out")
myfile = io.open("pred_formatted", "w")

for k = 1, #pred do
    myfile:write(pred[k][1] .. "\t[")
    for j = 1, pred[k][2]:size()[2] do
    	myfile:write(pred[k][2][1][j] .. ", ")
    end
    myfile:write("]\n")
end