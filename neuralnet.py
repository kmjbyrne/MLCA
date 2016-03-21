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

	report_data = {}
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
	for x in regressor_data:
		item={}
		item['index'] = str(i)
		item['prediction'] = round(x, 2)
		item['base'] = standard_data.target[i]
		item['probability_differential'] = abs(float(x + 1.0) - float(standard_data.target[i] + 1.0))
		i = i + 1
		report_set.append(item)

	report_data['regressor_results'] = report_set

	test_data_results = classifier_net.predictData(test_set.data)

	test_report_set = []
	for x in test_data_results:
		item = {}
		item['value'] = x
		test_report_set.append(item)

	i = 0
	report_data['regressor_results'] = test_report_set
	test_data_results = regression_net.predictData(test_set.data)

	ratios = []
	probability_report_set = []
	for x in regressor_data:
		item={}
		item['index'] = str(i)
		item['prediction'] = round(x, 2)
		item['base'] = data[i]
		item['probability_differential'] = round(abs(x - data[i]), 2)
		i = i + 1
		probability_report_set.append(item)

	#Determine predictions for actual test data
	class_test_results = classifier_net.predictData(test_set.data)
	range_test_results = regression_net.predictData(test_set.data)


	i = 0
	classifier_data = []
	#Iterate through the test set
	#Take classifier values
	#Take regressor values
	#Compare difference
	for x in class_test_results:
		item={}
		item['index'] = str(i)
		item['class_value'] = int(x)
		item['range_value'] = round(range_test_results[i], 4)
		item['deviation'] = round(abs(x - range_test_results[i]), 2)
		i = i + 1
		classifier_data.append(item)

	print(report_data)
	return render_template("report.html", regressor_data=report_set, data=classifier_data)

if __name__ == "__main__":
	app.run(debug=True)