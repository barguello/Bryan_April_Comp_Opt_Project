import scipy
import scipy.linalg

class AffineSpace(object):
    def __init__(self, vlist, offset = 0 ):
        """Defines an affine space
         
        vlist -- a list of vectors [[vec]], each in R^n
        offest -- some offset in R^n, [c1,c2..,cn]"""
        self.vlist = scipy.array(vlist)
        self.offset = scipy.array([offset])

    def getLocalCoords(self, p):
        """Return the local coordinates of a point p"""
        p = scipy.array(p)  - self.offset
        u,lam,vt = scipy.linalg.svd( self.vlist.T, full_matrices=False )
        laminv = scipy.diag(1.0/lam)
        return    vt.T.dot(  laminv.dot( u.T.dot( p.T ) ) ).T

    def getProjection(self, p):
        """Return the projection of point p"""
        p = scipy.array(p)
        return self.getLocalCoords(p).dot( self.vlist) + self.offset

    def isInSpace(self, p):
        """Return true if the point p is in the affine space, false otherwise."""
        p = scipy.array(p)
        close = scipy.isclose(self.getProjection(p),  p)
        inSpace = scipy.all(close, axis=1)
        return scipy.expand_dims(inSpace, 1)

if __name__ == '__main__':
    import affine_space
    s = affine_space.AffineSpace([[1,0,0],[0,1,0]], offset=[0,0,1])
    print s.getProjection( [[1,1,1], [10,10,10]])
    print s.isInSpace( [[1,1,1], [10,10,10], [1,1,0]])
