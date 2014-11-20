import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn import cross_validation
import pdb

#read in data and clean it
df = pandas.read_csv('school_data.csv')
df = df[~(df.ms_reading == '.')]
df = df[~(df.ms_math == '.')]
df.ms_math = df.ms_math.convert_objects(convert_numeric = True)
df.ms_reading = df.ms_reading.convert_objects(convert_numeric = True)
df = df[(df['ms_reading'] >= 0) & (df['ms_reading'] <= 100)]
df = df[(df['ms_math'] >= 0) & (df['ms_math'] <= 100)]

#add in drop-out rate
df = df[~(df.dropout_rate == '.')]
df.dropout_rate = df.dropout_rate.convert_objects(convert_numeric = True)
df = df[(df['dropout_rate'] >= 0) & (df['dropout_rate'] <= 100)]
#create indicator for exemplary schools
df = df[(df.rating == 'E') | (df.rating == 'L')]
df['indicator'] = 0
df.loc[ df.rating == 'E', 'indicator'] = 1

#store math, reading scores, drop-out rate as datapoints
x = df.ms_reading.values
x = x.tolist()
y = df.ms_math.values
y = y.tolist()
X = df.as_matrix(columns=['ms_reading','ms_math', 'dropout_rate'])
Y = df['indicator'].values

# fit the model
clf = svm.SVC(kernel='linear')
scores = cross_validation.cross_val_score(clf, X,Y, cv = 10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf.fit(X, Y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
min_x = min(x)
max_x = max(x)
xx = np.linspace(min_x, max_x, 100)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
#plt.plot(xx, yy, 'k-')
#plt.plot(xx, yy_down, 'k--')
#plt.plot(xx, yy_up, 'k--')

#plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')
#plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

#plt.axis('tight')
#plt.show()


# get the separating 3d hyperplane
w = clf.coef_[0]
min_x = min(x)
max_x = max(x)
min_y = min(y)
max_y = max(y)
xx = np.arange(min_x, max_x, .5)
yy = np.arange(min_y, max_y, .5)

# plot the plane and the points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xx, yy = np.meshgrid(xx, yy)
zz = -w[0]/w[2]*xx - w[1]/w[2]*yy - (clf.intercept_[0])/w[2]

surf = ax.plot_surface(xx,yy,zz, rstride=1, cstride=1, color = 'y', linewidth=0, antialiased=False)
ax.scatter(X[:,0], X[:,1], X[:,2], c=Y, cmap=plt.cm.seismic)

plt.axis('tight')
plt.show()





