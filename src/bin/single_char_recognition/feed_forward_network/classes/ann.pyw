import pickle

# Ann class

class Ann(object):

	# Function that pickles a neural network class instance into a binary file
	@staticmethod
	def pickle(ann, destination):
		with open(destination, "wb") as f:
			pickle.dump(ann, f, pickle.HIGHEST_PROTOCOL)

	# Function that unpickles the given ANN pickled file
	@staticmethod
	def unpickle(source):
		with open(source, "rb") as f:
			ann = pickle.load(f)
		return ann