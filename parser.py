import csv

class DataItem:

	def __init__(self, a,b,c,d,e,f,g,h,i,j,k,l,m,n):
		self.survival = a
		self.still_alive = b
		self.age_of_attack = c
		self.pericardial_effusion = d
		self.fractional_shortening = e
		self.e_point_septal_seperation = f
		self.left_ventricular_end_diastolic_dimension = g
		self.wall_motion_score = h
		self.wall_motion_index = i
		self.mult = j
		self.name = k
		self.group = l
		self.alive_after_year = m


def loadInputData(filename):
	with open(filename, 'rb') as data_file:
		loaded = csv.reader(data_file, delimiter =',')
		memory_data = []
		for row in loaded:
			obj = DataItem(row[0], row[1], row[2], row[3], 
				row[4], row[5], row[6], row[7], row[8], 
				row[9], row[10], row[11], row[12], "")
			memory_data.append(obj)
		
		for x in memory_data:
			print(x.survival)