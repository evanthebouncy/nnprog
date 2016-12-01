def softmax(vec):
  self_vals = [2**v for v in vec]
  total = sum(self_vals)
  return [v / total for v in self_vals]


