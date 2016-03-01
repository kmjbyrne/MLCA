from sknn.mlp import Regressor, Layer
from sklearn.datasets import load_boston
import numpy
import parser

boston = load_boston()
#
class NeuralNet():
	def __init__(self):
		self.nn = Regressor(
			layers=[
				Layer("Rectifier", units =100),
				Layer("Sigmoid", units = 28),
				Layer("Linear")],
			learning_rate = 0.02,
			n_iter = 10)

	def train(self):
		pass

class DataArray():
	pass



if __name__ == "__main__":
	data = parser.load_echo_data('data/data.csv')
	print(data)