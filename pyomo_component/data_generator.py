import numpy 
import scipy
import pdb
import matplotlib.pyplot

# create linearly separable points in the plane
def generate_linearly_separable_points(num = 40):
	numpy.random.seed(20)
	X1 = numpy.random.randn(num, 2) - [2, 2]
	X2 = numpy.random.randn(num, 2) + [2, 2]
	X = numpy.r_[X1,X2]
	Y = [-1] * num + [1] * num
	return X1,X2,X,Y

# create elliptically separable points in the plane
def generate_elliptically_separable_points(a,b, num = 40):
	#generate points inside of the ellipse
	pi = numpy.pi
	angle = 2*pi*scipy.rand(num)
	scale = numpy.array([a,b])
	X1 = scale * scipy.rand(num,2) * numpy.array([numpy.cos(angle), numpy.sin(angle)]).T
	Y = [-1] * num + [1] * num
	#generate points outside of the ellipse
	count = 0
	X2 = numpy.array([[a+1, b+1]])
	while count < num-1:
		rand_point = 2*scale*scipy.rand(2) - scale
		if (rand_point[0]/float(scale[0]))**2 + (rand_point[1]/float(scale[1]))**2 > 1:
			X2 = numpy.append(X2,numpy.array([rand_point]), axis = 0)
			count += 1
	return X1,X2, numpy.r_[X1,X2],Y

# create polynomially softly-separable points
def generate_polynomially_separable_points():
	X1 = numpy.c_[(.4, -.7),
	          (-1.5, -1),
	          (-1.4, -.9),
                  (-1.3, -1.2),
		  (-1.1, -.2),
	          (-1.2, -.4),
                  (-.5, 1.2),
                  (-1.5, 2.1),
                  (1, 1)].T
	X2 = numpy.c_[
                  (1.3, .8),
		  (1.2, .5),
                  (.2, -2),
	          (.5, -2.4),
                  (.2, -2.3),
                  (0, -2.7),
                 (1.3, 2.1)].T
	Y = [0] * 8 + [1] * 8
	return X1,X2, numpy.r_[X1,X2],Y

def plot_test(f):
	X1, X2, X, Y = generate_linearly_separable_points(40)
	matplotlib.pyplot.scatter(X1[:,0], X1[:,1], color = 'r')
	matplotlib.pyplot.scatter(X2[:,0], X2[:,1], color = 'b')
	matplotlib.pyplot.show()
	return

def elliptical_test():
	X1, X2, X, Y = generate_elliptically_separated_points(2,10,40)
	matplotlib.pyplot.scatter(X1[:,0], X1[:,1], color = 'r')
	matplotlib.pyplot.scatter(X2[:,0], X2[:,1], color = 'b')
	matplotlib.pyplot.show()
	return

def polynomial_test():
	X1, X2, X, Y =  generate_polynomially_separated_points()
        matplotlib.pyplot.scatter(X1[:,0], X1[:,1], color = 'r')
	matplotlib.pyplot.scatter(X2[:,0], X2[:,1], color = 'b')
	matplotlib.pyplot.show()
	return

