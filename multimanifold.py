from math import floor, ceil, log, sqrt
from random import randint
import numpy as np
from scipy.spatial.distance import pdist
import cvxpy as cp
from types import SimpleNamespace
from sklearn.linear_model import LogisticRegression
from typing import List

class MultiManifold:
    
    def __init__(self, sigma = 0.2, kfac = 0.5, Mfac = 1, 
                 neighfac = 3, mahalfac = 1, afac = 1.25, bfac = 1.25):
        self.sigma = sigma
        self.kfac = kfac
        self.Mfac = Mfac
        self.neighfac = neighfac
        self.mahalfac = mahalfac
        self.afac = afac
        self.bfac = bfac
        self.k = None
        self.m = None
        self.classifiers = None
        
        self.scratchpad = SimpleNamespace()
        self.scratchpad.XL = None
        self.scratchpad.y = None
        self.scratchpad.XU = None
        self.scratchpad.centroids = None
    
    def __distmatrix(self, X):
        dim = len(X)
        
        dist = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                pair = np.vstack((X[i], X[j]))
                dist[i][j] = pdist(pair)[0]
                
        return dist
    
    def __laplacian(self, W):
        dim = len(W)
        diag = np.zeros(dim, dim)
        for i in range(dim):
            sumi = 0
            for j in range(dim):
                sumi += W[i][j]
            diag[i][i] = sumi
            
        L = diag - W
        return L
    
    def __eigenize(self, L):
        eigenval, eigenvec = np.linalg.eig(L)
        eigensorted = np.argsort(eigenval)
        V = np.array(eigenvec(eigensorted[0]))
        for i in range(1, self.k):
            V = np.vstack((V, eigenvec(eigensorted[i])))
            
        return V
    
    def __cluster(self, V, a, b):
        n = len(self.scratchpad.XL)
        m = self.m
        k = self.k
        
        T = cp.Variable((n+m, k), boolean=True)
        c = cp.Variable((n+m, k))
        
        objective = 0.
        for i in range(n+m):
            for h in range(k):
                objective += T[i][h]((V[i] - c[h]) ** 2)
                
        constraints = []
        for i in range(n+m):
            constraints.append(sum([ T[i][h] for h in range(k) ]) == 1)
            
        for h in range(k):
            constraints.append(sum([ T[i][h] for i in range(n) ]) >= a)
            
        for h in range(k):
            constraints.append(sum([ T[i][h] for i in range(n, n+m) ]) >= b)
            
        prob = cp.Problem(objective, constraints)
        prob.solve()       
        
        return T.value, c.value
                
    def _base_classify(self) -> None:
        if (self.classifiers is None):
            self.classifiers = [ LogisticRegression() for i in range(self.k) ]
    
    # XL, y, XU are numpy arrays 
    # class labels are assumed to be 0,1, .., nclasses-1                 
    def fit(self, XL, y, XU):
        
        self.scratchpad.XL = XL
        self.scratchpad.y = y
        self.scratchpad.XU = XU
        
        n = len(XL)
        M = len(XU)
        D = np.shape(XL)[1]
        
        k = ceil(self.kfac * log(n))
        self.k = k
        m = ceil(self.Mfac * M * 1. / log(M))
        self.m = m
        
        neigh = floor(self.neighfac * log(M))
        a = floor(self.afac * n * 1. / (log(n)) ** 2)
        b = floor(self.bfac * m * 1. / (log(n)) ** 2)
        
        # euclidean distances
        points = np.vstack((XL, XU))
        dist = self.__distmatrix(points)
        udist = dist[n:n+M, n:n+M]
        u0 = randint(0, M)
        usel = [ u0 ]
        uneigh = {} # dictionary of nearest points (euclidean)
        
        i = 0
        while len(usel) < m:
            alldist = udist[usel[i]]
            pick = sorted(range(len(alldist)), 
                                  key=lambda r: alldist[r])[:neigh]
            uneigh[usel[i]] = pick
            for j in range(len(pick)):
                usel.insert(i+1+j, pick)
            i += 1
        # for some points in usel, nearest neighbors are not yet computed
        for uidx in usel:
            if not uidx in uneigh:
                alldist = udist[usel[uidx]]
                pick = sorted(range(len(alldist)), 
                              key=lambda r: alldist[r])[:neigh]
                uneigh[usel[i]] = pick
                
        ucov = []
        for i in range(len(usel)):
            nearest = uneigh[i]
            neighbors = [ XU[j] for j in nearest ]
            mu = np.mean(np.array(neighbors), axis=0)
            cov = 0.
            for point in neighbors:
                cov += np.dot((point - mu), (point - mu).T) * (1. / (neigh -1))
            ucov.append(cov)
            
        ldist = dist[0:n, 0:n+M]
        lcov = []
        for i in range(n):
            alldist = ldist[i]
            pick = sorted(range(len(alldist)), 
                          key=lambda r: alldist[r])[:neigh]
            neighbors = []
            for j in range(len(pick)):
                if j < n:
                    neighbors.append(XL[j])
                else:
                    neighbors.append(XU[j-n])
            mu = np.mean(np.array(neighbors), axis=0)
            cov = 0.
            for point in neighbors:
                cov += np.dot((point - mu), (point - mu).T) * (1. / (neigh -1))
            lcov.append(cov)
        
        unlab = np.array([ XU[i] for i in usel ])
        selX = np.vstack(XL, unlab)
        # All points selected - both labeled and unlabeled
        selidx = [ i for i in range(n) ]
        
        for i in range(len(usel)):
            selidx.append(n + usel[i])
        selcov = lcov.extend(ucov)
        
        nmahal = self.mahalfac * log(n + m)
        
        mahaldict = {}
        for i in range(n+m):
            mahali = []
            x = selX[i]
            for j in range(n+m):
                xprime = selX[j]
                covi = selcov[i]
                mahali.append(np.dot(np.dot((x - xprime), covi), (x - xprime)))
            mahalnear = sorted(range(len(mahali)), 
                                  key=lambda r: mahali[r])[:nmahal]
            mahaldict[i] = mahalnear
            
        W = np.zeros(n+m, n+m)
        for i in range(n+m):
            Sigma_i = selcov[i]
            near = mahaldict[i]
            for j in near:
                Sigma_j = selcov[j]
                di = np.linalg.det(Sigma_i)
                dj = np.linalg.det(Sigma_j)
                dij = np.linalg.det(Sigma_i + Sigma_j)
                if (W[j][i] != 0):
                    W[i][j] = W[j][i]
                else:
                    hellinger = sqrt(1 - (2 ** (D * 0.5)) * 
                                 ((di ** 0.25) * (dj ** 0.25) / (dij ** 0.5)))
                    W[i][j] = (hellinger ** 2) / (2 * self.sigma ** 2)
                    
        L = self.__laplacian(W)
        V = self.eigenize(L)
        T, c = self.__cluster(V, a, b)
        
        Xclustered = [ [] for i in range(k) ]
        yclustered = [ [] for i in range(k) ]
        for i in range(n): # only labeled examples
            cl = T[i].index(1)
            Xclustered[cl].append(XL[selX[i]])
            yclustered[cl].append(y[selX[i]])
            
        self.scratchpad.centroids = c
        
        self._base_classify()
        for i in range(self.k):
            self.classifiers[i].fit(np.array(Xclustered[i]), 
                                    np.array(yclustered[i]))
            
        print('Training Completed')
        
    def predict_proba(self, examples) -> List[float]:
        if not self.classifiers:
            raise ValueError('Model not yet trained.')
        
        predicted = []
        for X in examples:
            distances = []
            for i in range(self.k):
                pair = np.vstack((X, self.scratchpad.centroids[i]))
                distances.append(pdist(pair)[0])
            c = distances.index(min(distances))
            proba = self.classifiers[c].predict_proba([X])
            predicted.append(proba)
            
        return predicted
        
    def predict(self, examples) -> List[int]:
        if not self.classifiers:
            raise ValueError('Model not yet trained.')
        
        predicted = []
        for X in examples:
            proba = self.predict_proba([X])[0]
            c = proba.index(min(proba))
            predicted.append(c)
            
        return predicted
            
# end of class MultiManifold