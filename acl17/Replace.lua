local Replace, parent = torch.class("nn.Replace", "nn.Module")

function Replace:__init(original, replacement)
  parent.__init(self)
  self.original = original
  self.replacement = replacement
end

function Replace:updateOutput(input)
  local mask
  if self.original == "nan" then
    mask = torch.ne(input, input)
  else
    mask = torch.eq(input, self.original)
  end
  self.output = input.new(input:size()):copy(input)
  self.output:maskedFill(mask, self.replacement)
  return self.output
end

function Replace:updateGradInput(input, gradOutput)
  local mask
  if self.original == "nan" then
    mask = torch.ne(input, input)
  else
    mask = torch.eq(input, self.original)
  end
  self.gradInput = gradOutput.new(gradOutput:size()):copy(gradOutput)
  self.gradInput:maskedFill(mask, 0)
  return self.gradInput
end
