import numpy as np
from math import exp
from types import SimpleNamespace
from scipy.spatial.distance import pdist
import networkx as nx

class RegularizationManifold:
    
    def __init__(self, nneigh = 10, neigen = 5):
        self.nneigh = nneigh
        self.neigen = neigen
        
        self.scratchpad = SimpleNamespace()
        self.scratchpad.XL = None
        self.scratchpad.y = None
        self.scratchpad.XU = None
        self.scratchpad.dist = None
        self.scratchpad.weights = None
    
    def __distmatrix(self):
        if self.scratchpad.dist is not None:
            return self.scratchpad.dist
        
        XL = self.scratchpad.XL
        XU = self.scratchpad.XU
        
        points = np.vstack((XL, XU))
        dim = len(XL) + len(XU)
        dist = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                pair = np.vstack((points[i], points[j]))
                dist[i][j] = pdist(pair)[0]
                
        self.scratchpad.dist = dist
        return dist
    
    def __calcweights(self):
        
        if self.scratchpad.weights is not None:
            return self.scratchpad.weights
        
        XL = self.scratchpad.XL
        XU = self.scratchpad.XU
        
        # dist = self.__distmatrix(XL, XU)
        dim = len(XL) + len(XU)
        
        weights = np.zeros(dim, dim)
        for i in range(dim):
            tmpdist = [ (self.scratchpad.dist[i][j], i, j) for j in range(dim) ]
            tmpdist.sort()
            for r in range(self.nneigh):
                weights[tmpdist[r][1], tmpdist[r][2]]= 1
        
        self.scratchpad.weights = weights
        return weights
    
    # Alternatively we may implement gradient descent
    def __coefficients(self, Elab):
        y = self.scratchpad.y
        return np.dot(np.dot((np.invert(np.dot(Elab.T, Elab))), (Elab.T)), y.T)
    
    # Class labels are +1 and -1
    def fit(self, XL, y, XU):
        self.scratchpad.XL = XL
        self.scratchpad.y = y
        self.scratchpad.XU = XU
        
        W = self.__calcweights()
        
        l = len(XL)
        u = len(XU)
        D = np.zeros(l+u, l+u)
        for i in range(l+u):
            sumi = 0
            for j in range(l+u):
                sumi += W[i][j]
            D[i][i] = sumi
            
        #L = W - D
        L = D - W
        
        eigenval, eigenvec = np.linalg.eig(L)
        ev = [ (eigenval, i) for i in range(len(eigenval)) ]
        evsorted = ev.sort()
        
        evec = []
        for i in range(self.neigen):
            j = evsorted[i][1]
            evec.append(eigenvec[j])
            
        E = np.array(evec)
        
        Elab = E[:l, :l]
        a = self.__coefficients(Elab)
        
        classes = [ 0 ] * u
        for j in range(u):
            prod = np.dot(E[j], a)
            if prod > 0:
                classes[j] = 1
            else:
                classes[j] = -1
                
        self.scratchpad.newlabels = classes
        
        return classes
    
# end of class RegularizationManifold