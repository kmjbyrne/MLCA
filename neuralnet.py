from flask import Flask,render_template,jsonify,url_for,request,session
from sknn.mlp import Classifier, Layer, Regressor
import numpy
import parser

import sys
import logging

logging.basicConfig(
	format= "%(message)s",
	level = logging.DEBUG,
	stream=sys.stdout)

app = Flask(__name__)
data = None

class ClassifierNeuralNet():
	def __init__(self):
		self.nn = Classifier(
			layers=[
				Layer("Sigmoid", units =100),
				Layer("Softmax")],
			learning_rate = 0.001,
			n_iter = 200)

	def train(self):
		data = parser.load_echo_data('data/training_data.csv')
		self.nn.fit(data.data, data.target)

	def predictData(self, data):
		return self.nn.predict(data)

class RegressorNeuralNet():
	def __init__(self):
		self.nn = Regressor(
		    layers=[
		        Layer("Sigmoid", units=100),
		        Layer("Sigmoid", units=47),
		        Layer("Linear")],
		    learning_rate=0.02,
		    n_iter=200)

	def train(self):
		data = parser.load_echo_data('data/training_data.csv')
		self.nn.fit(data.data, data.target)

	def predictData(self, data):
		return self.nn.predict(data)

class DataArray():
	pass


@app.route("/", methods=['GET'])
def report():

	data= None
	predicted_data = None
	standard_data = parser.load_echo_data('data/training_data.csv')
	input_data = standard_data.data

	test_set = parser.loadTestData('data/data.csv')

	try:
		classifier_net = ClassifierNeuralNet()
		regression_net = RegressorNeuralNet()
		classifier_net.train()
		regression_net.train()

	except Exception as e:
		print(e)

	try:
		data = classifier_net.predictData(input_data)
		regressor_data = regression_net.predictData(input_data)
	except Exception as e:
		print(e)

	report_set = []
	i = 0
	for x in data:
		item={}
		item['index'] = str(i)
		item['prediction'] = x
		item['base'] = standard_data.target[i]
		item['probability_differential'] = abs(float(x + 1.0) - float(standard_data.target[i] + 1.0))
		i = i + 1
		report_set.append(item)

	test_data_results = classifier_net.predictData(test_set.data)

	test_report_set = []
	for x in test_data_results:
		item = {}
		item['value'] = x
		test_report_set.append(item)

	i = 0

	#test_data_results = regression_net.predictData(test_set.data)

	ratios = []
	probability_report_set = []
	for x in regressor_data:
		item={}
		item['index'] = str(i)
		item['prediction'] = data[i]
		item['chance'] = round(x, 2)
		item['base'] = data[i]
		item['probability_differential'] = round(abs(x - data[i]), 2)
		i = i + 1
		probability_report_set.append(item)

	info={}
	info['classifier'] = vars(classifier_net.nn)
	info['regression'] = vars(classifier_net.nn)
	#print(classifier_net.nn.get_parameters())
	#print(regression_net.nn.get_parameters())7

	return render_template("report.html", data=report_set, predict_data = test_report_set, probability_data=probability_report_set, info=info)

if __name__ == "__main__":
	app.run(debug=True)