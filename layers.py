class Matmul:
	def __init__(self, w):
		self.params = [w]
		self.grads = [np.zeros_like(w)]
		self.x = None

	def forward(self, x):
		w, = self.params
		out = np.dot(x, w)
		self.x = x
		return out

	def backward(self, dout):
		w, = self.params
		dx = np.dot(dout, w.T)
		dw = np.dot(self.x.T, dout)
		self.grads[0][...] = dw
		return (dx)