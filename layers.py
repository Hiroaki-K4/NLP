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


class Sigmoid:
	def __init__(self):
		self.params, self.grads = [], []
		self.out = None

	def forward(self, x):
		out = 1 / (1 + np.exp(-x))
		self.out = out
		return out

	def backward(self, dout):
		dx = dout * (1.0 - self.out) * self.out
		return dx


class Affine:
	def __init__(self, w, b):
		self.params = [w, b]
		self.grads = [np.zeros_like(w), np.zeros_like(b)]
		self.x = None

	def forward(self, x):
		w, b = self.params
		out = np.dot(x, w) + b
		self.x = x
		return out

	def backward(self, dout):
		w, b = self.params
		dx = np.dot(dout, w.T)
		dy = np.dot(self.x.T, dout)
		db = np.sum(dout, axis=0)

		self.grads[0][...] = dw
		self.grads[1][...] = db
		return dx