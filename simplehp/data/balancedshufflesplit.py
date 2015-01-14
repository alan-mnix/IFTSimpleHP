import sklearn
import sklearn.cross_validation
import sklearn.preprocessing
import numpy

class BalancedShuffleSplit(object):
	"""docstring for BalancedShuffleSplit"""
	def __init__(self, y, n = 5, train_size = 0.25, random_state = numpy.random.RandomState()):
		super(BalancedShuffleSplit, self).__init__()
		self._nsamples = len(y)
		self._n = n
		self._rng = random_state
		if 0 < train_size < 1:
			self._trainsize = int(train_size*self._nsamples)
		else:
			self._trainsize = train_size
		
		self._labelencoder = sklearn.preprocessing.LabelEncoder()
		self._y = self._labelencoder.fit_transform(y)#make the labels in range [0,c), where c is number of labels
		self._nclasses = max(self._y)+1
		self._buckets = [numpy.where(self._y==i)[0] for i in numpy.unique(self._y)]

		self._samplesperclass = self._trainsize/self._nclasses
		self._excess = self._trainsize%self._nclasses

		classes_not_ok = []
		for i in numpy.unique(self._y):
			
			if not (sum(self._y==i)>(self._samplesperclass+self._excess)):
				classes_not_ok.append(i)

		#print self._trainsize
		#print self._samplesperclass+self._excess

		if len(classes_not_ok)>0:
			raise Exception('Need at least %d samples for classes = %s'%(self._samplesperclass+self._excess+1, self._labelencoder.inverse_transform(classes_not_ok)))

	def __iter__(self):
		for i in xrange(self._n):
			yield self._generateBalancedShuffleSplit()


	def _generateBalancedShuffleSplit(self):
		train = numpy.ones(0)
		test = numpy.ones(0)
		idx = 0
		for i in self._buckets:
			self._rng.shuffle(i)
			if idx<self._excess:
				trainsize = self._samplesperclass+1
			else:
				trainsize = self._samplesperclass
			#print trainsize
			train = numpy.append(train, i[:trainsize])
			test = numpy.append(test, i[trainsize:])
			idx+=1

		# print (train, test)
		return (train.astype(int), test.astype(int))
