import pandas
import numpy
import scipy
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn import cross_validation
from itertools import chain, combinations
import IPython.parallel
import affine_space
import pdb
from preprocessing import * 

def classify(feature_list, df, kernel = 'poly', plot = 'SVD', num_plots = 100):
	X = df.as_matrix(columns=feature_list)
	Y = df['classifier'].values
	# fit the model
	clf = svm.SVC(kernel=kernel, class_weight='auto' )
	scores = cross_validation.cross_val_score(clf, X,Y, cv = 10)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	clf.fit(X, Y)
	
	if plot == '2D':
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

	elif plot == 'SVD':
		Xbar = numpy.mean(X[X > 0], axis = 0)
		X = X - Xbar
		Cov = numpy.cov(X.T)
		eigvalues, eigvectors = numpy.linalg.eig(Cov)
		indices = eigvalues.argsort()[-2:][::-1]
		direction1 = eigvectors[:,indices[0]]
		direction2 = eigvectors[:,indices[1]]
		#U, s, Vh = numpy.linalg.svd(X.T)
		#direction1 = U[:,0]
		#direction2 = U[:,1]
		print direction1
		print direction2
		projection_space = affine_space.AffineSpace(numpy.array([direction1, direction2]))
		local_coords = numpy.array([projection_space.getLocalCoords(x) for x in X])

		x_min = numpy.min(X[:,0])
		x_max = numpy.max(X[:,0])
		y_min = numpy.min(X[:,1])
                y_max = numpy.max(X[:,1])
		
		matplotlib.pyplot.figure(1, figsize=(4, 3))
		matplotlib.pyplot.clf()

		matplotlib.pyplot.scatter(local_coords[:, 0], local_coords[:, 1], zorder=5, c=Y, cmap=matplotlib.pyplot.cm.spring)
		matplotlib.pyplot.xlim(x_min, x_max)
		matplotlib.pyplot.ylim(y_min, y_max)
		matplotlib.pyplot.show()

	elif plot == 'random':
		dimension = numpy.shape(X)[1]
		for i in range(num_plots):
			direction1 = scipy.rand(1,dimension)
			direction2 = scipy.rand(1, dimension)
			projection_space = affine_space.AffineSpace(numpy.array([direction1[0], direction2[0]]))
			local_coords = numpy.array([projection_space.getLocalCoords(x) for x in X])

			x_min = numpy.min(X[:,0])
			x_max = numpy.max(X[:,0])
			y_min = numpy.min(X[:,1])
                	y_max = numpy.max(X[:,1])
			
			matplotlib.pyplot.clf()
	
			matplotlib.pyplot.scatter(local_coords[:, 0], local_coords[:, 1], zorder=2, c=Y, cmap=matplotlib.pyplot.cm.bwr)
			matplotlib.pyplot.xlim(x_min, x_max)
			matplotlib.pyplot.ylim(y_min, y_max)
			matplotlib.pyplot.savefig('plot_' + str(i) + '.jpg')
		
	else:
		return

def powerset(iterable):
	"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def test_over_powerset(feature_list, df, kernel = 'poly', file_name = 'output.txt'):
	f = open(file_name, 'w')
	for sublist in powerset(feature_list):
		length = len(sublist)
		if length >= 2:
			X = df.as_matrix(columns = sublist)
			Y = df['classifier'].values
			#fit the model
			clf = svm.SVC(kernel = kernel, class_weight = 'auto')
			score = cross_validation.cross_val_score(clf, X, Y, cv = 10)
			f.write(str(sublist) + ": " + "Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2) + '\n')	
	f.close()

hs = pandas.read_csv('hs.csv')
#list1 = ['perLEP', 'teacherStudentRatio'] #84 +- 9
#list2 = ['perEconDisadv', 'perLEP', 'teacherStudentRatio'] #84 +- 9
list3 = ['teacherStudentRatio', 'perSPED']
list4 = ['perLEP', 'perSPED', 'perAtRisk', 'dropoutRate', 'perGT']
classify(feature_list, hs, kernel = 'linear', plot = 'random', num_plots = 100)
#test_over_powerset(list3, hs, kernel = 'poly', file_name = 'list3.txt')
