import pandas
import numpy
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn import cross_validation
from itertools import chain, combinations
import IPython.parallel
import pdb
#from preprocessing import *

def classify(feature_list, df, kernel = 'poly', plot = False):
	X = df.as_matrix(columns=feature_list)
	Y = df['classifier'].values
	# fit the model
	clf = svm.SVC(kernel=kernel, class_weight='auto' )
	scores = cross_validation.cross_val_score(clf, X,Y, cv = 10)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	
	if(plot):
		clf.fit(X, Y)
		x_min = numpy.min(X[:,0])
		x_max = numpy.max(X[:,0])
		y_min = numpy.min(X[:,1])
                y_max = numpy.max(X[:,1])

		matplotlib.pyplot.figure(1, figsize=(4, 3))
		matplotlib.pyplot.clf()

		matplotlib.pyplot.scatter(X[:, 0], X[:, 1], zorder=10, c=Y, cmap=matplotlib.pyplot.cm.hot)
		XX, YY = numpy.mgrid[x_min:x_max:600j, y_min:y_max:600j]
		Z = clf.decision_function(numpy.c_[XX.ravel(), YY.ravel()])
		Z = Z.reshape(XX.shape)

		matplotlib.pyplot.pcolormesh(XX, YY, Z > 0, cmap=matplotlib.pyplot.cm.Paired)
		matplotlib.pyplot.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
		                                levels=[-.5, 0, .5])
		matplotlib.pyplot.xlim(x_min, x_max)
		matplotlib.pyplot.ylim(y_min, y_max)
		matplotlib.pyplot.show()

def powerset(iterable):
	"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
	s = list(iterable)
	iterable_list = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
	return [list(element) for element in iterable_list]

def test_over_powerset(given_list, df, file_name = 'output.txt'):
	c = IPython.parallel.Client()
	lv = c.load_balanced_view()

	@IPython.parallel.require('preprocessing', 'sklearn', 'sklearn.svm', 'sklearn.cross_validation')
	def parallel_function(sublist):
		length = len(sublist)
		if length < 2:
			return -1
			
		X = preprocessing.df.as_matrix(columns = sublist)
		Y = preprocessing.df['classifier'].values
		clf = sklearn.svm.SVC(kernel = 'poly', class_weight = 'auto')
		return sklearn.cross_validation.cross_val_score(clf, X, Y, cv = 10)

	f = open(file_name, 'w')
	list_powerset = powerset(given_list)
	print list_powerset
	r = lv.map(parallel_function, list_powerset)
	print "parallelization done"
	print r
	scores = r.get()
	print scores
	for element in scores:
		f.write(str(element[0]) + ": " + "Accuracy: %0.2f (+/- %0.2f)" % (element[1].mean(), element[1].std() * 2) + '\n')	
	f.close()

hs = pandas.read_csv('hs.csv')
#list1 = ['perEconDisadv', 'perLEP', 'teacherStudentRatio', 'perHisp', 'perAA'] #88 +- 9
#list2 = ['perEconDisadv', 'perLEP', 'teacherStudentRatio', 'perSPED', 'perAtRisk', 'dropoutRate', 'perGT'] #91 +- 4 
list3 = ['perEconDisadv', 'perLEP', 'teacherStudentRatio']
list4 = ['perEconDisadv', 'perLEP']
#classify(list15, hs, kernel = 'poly', plot = True)
test_over_powerset(list4, hs, file_name = 'list4.txt')
