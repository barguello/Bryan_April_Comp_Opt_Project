import numpy
import scipy
import matplotlib.pyplot
from coopr.pyomo import *
import coopr.environ
from coopr.opt import SolverFactory
from sklearn import svm
from sklearn import cross_validation
import data_generator
import time
import pdb

#get data for classification
X1, X2, X, Y = data_generator.generate_linearly_separable_points(20)
data_size = numpy.shape(Y)[0]

#starting pyomo timer
start_time = time.time()

#initiate and setup model
model = ConcreteModel()
model.i = RangeSet(0, data_size -1, 1)

#create parameters and variables
def init_x(model, i):
	return X[i]

def init_y(model, i):
	return Y[i]

model.x = Param(model.i, initialize = init_x)
model.y = Param(model.i, initialize = init_y)
C = 5
model.alpha = Var(model.i, within = NonNegativeReals, bounds = (0, C))

#create constraints
def orth_constraint(model):
	return summation(model.y, model.alpha) == 0
model.orth = Constraint(rule = orth_constraint)

#create objective function
def kernel(x,y):
	max_index = len(numpy.shape(y))
	return numpy.sum(x*y, axis = max_index - 1)

def Obj(model):
	return summation(model.alpha) - 0.5*sum([ sum([model.alpha[i]*model.alpha[j]*model.y[i]*model.y[j]*kernel(model.x[i],model.x[j]) for i in model.i]) for j in model.i])

model.L = Objective(rule = Obj, sense = maximize)

#Create Solver
opt = SolverFactory('ipopt')

#Solve
results = opt.solve(model)
model.load(results)

#Ending pyomo timer
print("Pyomo took " + str(time.time() - start_time) + " seconds")

#Scatter original points
matplotlib.pyplot.figure(1, figsize=(4, 3))
matplotlib.pyplot.clf()
matplotlib.pyplot.scatter(X1[:,0], X1[:,1], color = 'r', zorder=10)
matplotlib.pyplot.scatter(X2[:,0], X2[:,1], color = 'b', zorder=10)

#Get a picture for the classifier
x_min = numpy.min(X[:,0])
x_max = numpy.max(X[:,0])
y_min = numpy.min(X[:,1])
y_max = numpy.max(X[:,1])

def decision_function(model, X):
	return sum([model.alpha[k].value*model.y[k]*kernel(model.x[k], X) for k in model.i])

XX, YY = numpy.mgrid[x_min:x_max:600j, y_min:y_max:600j]
Z = decision_function(model, numpy.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)

matplotlib.pyplot.pcolormesh(XX, YY, Z > 0, cmap=matplotlib.pyplot.cm.Paired)
matplotlib.pyplot.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
		                levels=[-.5, 0, .5])
matplotlib.pyplot.xlim(x_min, x_max)
matplotlib.pyplot.ylim(y_min, y_max)
matplotlib.pyplot.show()

#sci-kit SVM
start_time = time.time()
clf = svm.SVC(kernel='linear', class_weight='auto' )
clf.fit(X, Y)
print("Sci-kit took " + str(time.time() - start_time) + " seconds")
