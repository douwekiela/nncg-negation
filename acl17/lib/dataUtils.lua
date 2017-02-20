local tnt = require "torchnet"
local js = require "cjson"
local moses = require "moses"

function padTensor(length, tensor)
   local tensorLength = tensor:size(1)
   if tensorLength < length then
      do
	 return torch.cat(tensor, torch.zeros(length-tensorLength), 1)
      end
   else
      return torch.Tensor(tensor:size()):copy(tensor)
   end
end

function tabSlice(tab, start, endi)
   local tabOut = {}
   local endIdx = (endi < 0) and (#tab + endi + 1) or endi
   for i = 1,(endIdx-start+1) do
      tabOut[i] = tab[start + i - 1]
   end
   return tabOut
end

function padTable(tab, length)
   local tabLength = #tab
   local tabCopy = tnt.utils.table.clone(tab)
   for i=1,#tabCopy do
      tabCopy[i] = torch.Tensor{tabCopy[i]}
   end
   for i=1,(length-tabLength) do
      tabCopy[tabLength + i] = torch.Tensor{0}
   end
   return tabCopy
end


function loadFromJSON(filename, eval)
   dataFile = io.open(filename, "r")
   dataText = dataFile:read("*all")
   dataTable = js.decode(dataText)
   for i, example in pairs(dataTable)
   do
      example["input"] = torch.Tensor(example["input"])
      example["context"] = torch.Tensor(example["context"])
      example["output"] = torch.Tensor(example["output"])
      if eval then
	 for i, noise in pairs(example["noise"]) do
	    example["noise"][i] = torch.Tensor(noise)
	 end
      end
   end
   return dataTable
end

function tabGetn(n)
   return tnt.transform.tableapply(function(ex)
	 return ex[n]
   end)
end

function tabGetn2(n)
   return tnt.transform.tableapply(function(ex)
	 return ex[{{}, n}]
   end)
end
local tabLength = tnt.transform.tableapply(function(x) return x:size(1) end)

function padBatch(dataTable) 
   local maxLen = torch.Tensor(tabLength(dataTable)):max()
   local mergeData = tnt.utils.table.mergetensor(
      tnt.utils.table.foreach{
	 tbl = dataTable,
	 closure = moses.bind(padTensor, maxLen)}):transpose(1, 2)
   return mergeData
end

function padMerge(batchTable)
   local getInput = tabGetn(1)
   local getOutput = tabGetn(2)
   local getContext = tabGetn(3)
   local inputTable, contextTable = getInput(batchTable["input"]), getContext(batchTable["input"]) 
   local outputTable, targetTable = getOutput(batchTable["input"]), batchTable["target"]
   print("contexttable:", contextTable)
   local mergeContext = tnt.utils.table.mergetensor(contextTable)
   local mergeInput, mergeOutput, mergeTarget= padBatch(inputTable), padBatch(outputTable), padBatch(targetTable)
   return {input = {mergeInput,
		    mergeOutput,
		    mergeContext},
	   target = mergeTarget}
end

function mergeEval(cuda, batchTable)
   local getFirst = tabGetn(1)
   local getSecond = tabGetn(2)
   local getThird = tabGetn(3)
   local tabSliceOutput = tnt.transform.tableapply(function(tensor)
	 return tensor[{{1, -2}}]
   end)
   local tabSliceTarget = tnt.transform.tableapply(function(tensor)
	 return tensor[{{2, -1}}]
   end)
 
   local gold, noise  = getFirst(batchTable["input"]), getSecond(batchTable["input"])

   local batchGoldInput = padBatch(getFirst(gold))
   local batchGoldOutput = padBatch(tabSliceOutput(getSecond(gold)))
   local batchGoldTarget = padBatch(tabSliceTarget(getSecond(gold)))
   local batchGoldContext = tnt.utils.table.mergetensor(getThird(gold))

   local noiseInput, noiseNoise, noiseContext = getFirst(noise), getSecond(noise), getThird(noise)
   local batchNoiseInput = padBatch(noiseInput)
   local batchNoiseContext = tnt.utils.table.mergetensor(noiseContext)
   local batchNoiseNoise, batchNoiseTarget
   if type(noiseNoise[1]) == "table" then
      local lenNoise = #noiseNoise[1]
      batchNoiseNoise = {}
      batchNoiseTarget = {}
      for i=1,lenNoise do
	 local noises = tabGetn(i)(noiseNoise)
	 --print("noises", noises)
	 batchNoiseNoise[i] = padBatch(tabSliceOutput(noises))
	 batchNoiseTarget[i] = padBatch(tabSliceTarget(noises))
	 if cuda then
	    batchNoiseNoise[i],  batchNoiseTarget[i] = batchNoiseNoise[i]:cuda(),  batchNoiseTarget[i]:cuda()
	 end
      end
   else
      batchNoiseNoise = padBatch(tabSliceOutput(noiseNoise))
      batchNoiseTarget = padBatch(tabSliceTarget(noiseNoise))
      if cuda then
	 batchNoiseNoise = batchNoiseNoise:cuda()
	 batchNoiseTarget = batchNoiseTarget:cuda()
      end
   end
   local batchTarget = torch.cat(batchTable["target"]) 
   
   if cuda then
      batchNoiseInput = batchNoiseInput:cuda()
      batchNoiseContext = batchNoiseContext:cuda()
      batchGoldInput = batchGoldInput:cuda()
      batchGoldOutput = batchGoldOutput:cuda()
      batchGoldContext = batchGoldContext:cuda()
      batchGoldTarget = batchGoldTarget:cuda()
      batchTarget = batchTarget:cuda()
   end
   
   local batchNoise = {input = {batchNoiseInput, batchNoiseNoise, batchNoiseContext},
		       target = batchNoiseTarget}
   local batchGold = {input = {batchGoldInput, batchGoldOutput, batchGoldContext},
		      target = batchGoldTarget}
   
   return {
      input = {batchGold, batchNoise},
      target = batchTarget 
   }
end

function noiseId(idx, maxLim)
   local outIdx =  torch.random(1, maxLim)
   return (outIdx == idx) and noiseId(idx, maxLim) or outIdx
end

function loadDataset(outData, batchSize, cuda, eval)
   print("Loading data")
   local compare = function(v1, v2)
      return (v1["input"]:size(1) + v1["output"]:size(1)) < (v2["input"]:size(1) + v2["output"]:size(1))
   end
   outData = moses.select(outData, function(k,v)
			     return (v["input"]:nDimension() ~= 0) and (v["output"]:nDimension() ~= 0)
   end)
   local mergeFunc
   if eval then
      mergeFunc = moses.bind(mergeEval, cuda)
   else
      mergeFunc = moses.bind(mergeEval, cuda)
   end
   table.sort(outData, compare)
   outDataset = tnt.ShuffleDataset{
      dataset = tnt.BatchDataset{
	 batchsize = batchSize,
	 dataset = tnt.ListDataset{
	    list = torch.range(1, #outData):long(),
	    load = function(idx)
	       local out
	       if eval then
		  out = {
		     input = {{outData[idx]["input"], outData[idx]["output"], outData[idx]["context"]},
			{outData[idx]["input"], outData[idx]["noise"], outData[idx]["context"]}},
		     target = torch.Tensor({1})
		  }
	       else
		  local noiseIdx = noiseId(idx, #outData)
		  local gold = {outData[idx]["input"], outData[idx]["output"], outData[idx]["context"]}
		  local noise = {outData[idx]["input"], outData[noiseIdx]["output"], outData[idx]["context"]}
		  out = {
		     input = {gold, noise},
		     target = torch.Tensor({1})
		  }
	       end  
	       return out
	    end
	 },
	 merge = mergeFunc
      }
   }
   outDatasetIterator = tnt.DatasetIterator{dataset = outDataset}
   print("Done loading data")
   return outDatasetIterator
end
