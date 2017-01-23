-- Beam search for Elemental Research/RNN
-- Joost van Doorn <joost.vandoorn@student.uva.nl>
local cutorch = require "cutorch"
local BeamSearch = torch.class('BeamSearch')

function BeamSearch:__init(beamSize, maxLength, startSymbol, endSymbol, unknownSymbol)
   --parent.__init(self)
   self.beamSize = beamSize
   self.maxLength = maxLength
   self.startSymbol = startSymbol
   self.endSymbol = endSymbol
   self.unknownSymbol = unknownSymbol
end

function BeamSearch:search(dec, stateFunc, copyFunc)
   assert(dec.sharedClone, "Only use beam search with modules that support sharedClone")
   dec:forget()
   dec:remember('neither')
   local completeHypo = {}
   -- initialize bin
   local bin = {}
   local hypo0 = {y = {self.startSymbol}, cost = 0.0, prevState = {}}
   bin[0] = {hypo0}
   for t = 1, self.maxLength do
      collectgarbage()
      -- check the previous bin
      local prevBin = bin[t-1]
      local buffHypo = {}
      local prePrune = nil
      for _,hypo in pairs(prevBin) do
	 local y = hypo.y[#hypo.y] -- take the last element
	 y = torch.CudaTensor{y}:view(1,1)
	 -- decoder
	 copyFunc(hypo.prevState)
	 local out = dec:forward(y)
	 -- add to beam
	 local val, id = torch.sort(out[1], true)
	 if prePrune == nil then
	    assert(id:size(2)>=self.beamSize, "Beam size bigger than vocabulary")
	    prePrune = val[1][self.beamSize]
	 else
	    prePrune = math.max(prePrune, val[1][self.beamSize])
	 end
	 for j = 1,self.beamSize do
	    local yn = id[1][j]
	    local logp = val[1][j]
	    -- Preprune
	    if prePrune>logp then
	       break
	    end
	    local ys = {}
	    for _,v in pairs(hypo.y) do ys[#ys+1] = v end
	    ys[#ys+1] = yn
	    local newHypo = {y = ys, prevState = {}, cost = hypo.cost + logp}
	    if yn ~= self.endSymbol then
	       newHypo.prevState = stateFunc()
	    end
	    if yn == self.endSymbol then
	       newHypo.prevState = nil
	       completeHypo[#completeHypo + 1] = newHypo
	    elseif yn ~= self.unknownSymbol then
	       buffHypo[#buffHypo + 1] = newHypo
	    end
	 end
	 hypo.prevState = nil
      end
      -- pruning
      table.sort(buffHypo, function(h1,h2)
		    return h1.cost > h2.cost
      end)
      if #buffHypo == 0 then break end
      bin[t] = {}
      for j = 1,math.min(self.beamSize,#buffHypo) do
	 bin[t][j] = buffHypo[j]
      end
   end
   local last_bin = bin[#bin]
   for _,hypo in pairs(last_bin) do
      hypo.prevState = nil -- Cleanup
      table.insert(completeHypo, hypo)
   end

   table.sort(completeHypo, function(h1,h2)
		 return h1.cost > h2.cost
   end)
   local best_hypo = completeHypo[1]
   return best_hypo
end

local GreedySearch = torch.class("GreedySearch")

function GreedySearch:__init(decoder, startSymbol, endSymbol)
   self.decoder = decoder
   self.startIdx = startSymbol
   self.endIdx = endSymbol
end

function GreedySearch:search(input, maxLength)
   local i = 1
   local symbol = self.startSymbol
   local sequence = {}
   local encoding = self.decoder:encode(input)

   while (symbol ~= self.endSymbol) and (i < maxLength) do
      table.insert(sequence, symbol)
      local predOut = self.decoder:decode(encoding, {symbol})
      local top, idx = torch.topk(predOut, 1, 1, true)
      symbol = ""
      i = i+1
   end
   return sequence
end

-- print("Beam search test")
-- local encOut = enc:forward(encInSeq)
-- forwardConnect(encLSTM, decLSTM)
-- local bs = BeamSearch(12, 50, 5, 6, 10)
-- local function stateFunc()
--   -- userPrevOutput
--   -- cell
--   return {decLSTM.output[1]:clone(), decLSTM.cell[1]:clone()}
-- end
-- local function copyFunc(state)
--   decLSTM.userPrevOutput = state[1]
--   decLSTM.userPrevCell = state[2]
-- end
-- print(bs:search(dec, stateFunc, copyFunc))
-- local encOut = enc:forward(encInSeq)
-- forwardConnect(encLSTM, decLSTM)
-- bs.beamSize = 1
-- print(bs:search(dec, stateFunc, copyFunc))
-- local encOut = enc:forward(encInSeq)
-- forwardConnect(encLSTM, decLSTM)
-- bs.beamSize = 3
-- print(bs:search(dec, stateFunc, copyFunc))
