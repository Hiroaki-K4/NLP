import numpy as np


class Sigmoid:
	def __init__(self):
		self.params = []

	def forward(self, x):
		return 1 / (1 + np.exp(-x))


class Affine:
	def __init__(self, w, b):
		self.params = [w, b]

	def forward(self, x):
		w, b = self.params
		out = np.dot(x, w) + b
		return out


class TwoLayerNet:
	def __init__(self, input_size, hidden_size, output_size):
		I, H, O = input_size, hidden_size, output_size
		w1 = np.random.randn(I, H)
		b1 = np.random.randn(H)
		w2 = np.random.randn(H, O)
		b2 = np.random.randn(O)

		self.layers = [
			Affine(w1, b1),
			Sigmoid(),
			Affine(w2, b2)
		]

		self.params = []
		for layer in self.layers:
			self.params += layer.params
			print("params: ", self.params)

	def predict(self, x):
		for layer in self.layers:
			x = layer.forward(x)
		return (x)


if __name__ == '__main__':
    x = np.random.randn(10, 2)
    print("x:",x)
    model = TwoLayerNet(2, 4, 3)
    s = model.predict(x)
    print(s)