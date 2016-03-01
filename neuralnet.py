from flask import Flask,render_template,jsonify,url_for,request,session
from sknn.mlp import Regressor, Classifier, Layer
import numpy
import parser

app = Flask(__name__)
data = None

class NeuralNet():
	def __init__(self):
		self.nn = Regressor(
			layers=[
				Layer("Rectifier", units =100),
				Layer("Sigmoid", units = 28),
				Layer("Linear")],
			learning_rate = 0.02,
			n_iter = 25000)

	def train(self):
		data = parser.load_echo_data('data/training_data.csv')
		self.nn.fit(data.data, data.target)
		return self.nn.predict(data.data)

class DataArray():
	pass


@app.route("/", methods=['GET'])
def report():
	predicted_data = None
	try:
		net = NeuralNet()
		predicted_data = net.train()
	except Exception as e:
		print(e)

	for element in predicted_data:
		print(element)

	print(predicted_data)
	return render_template("report.html", data=predicted_data)

if __name__ == "__main__":
	app.run()