import csv
import numpy as np

class DataItem(dict):

	def __init__(self, **kwargs):
		dict.__init__(self, kwargs)

	def __setattr__(self, key, value):
		self[key]= value

	def __getattr__(self, key):
		try:
			return self[key]
		except KeyError as e:
			print(e)

	def __setstate__(self, state):
		pass

def load_echo_data(filename):
	"""
	Load and return echo-cardiogram dataset (regression)

	===========================

	Sample data specs:


	===========================

	"""

	with open(filename, 'rb') as f:
		data_file = csv.reader(f)
		temp = next(data_file)
		n_samples = float(temp[0])
		n_features = float(temp[1])

		data = np.empty((n_samples, n_features))
		target = np.empty((n_samples,))
		temp = next(data_file) #names of features
		feature_names = np.array(temp)

		for i, d in enumerate(data_file):
			data[i] = np.asarray(d[:-1], dtype=np.float)
			target[i] = np.asarray(d[-1], dtype=np.float)

	return DataItem(data=data,
					target=target,
					feature_names=feature_names[:-1],
					DESCR="echo-cardiogram")