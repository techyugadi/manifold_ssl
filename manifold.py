import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import load_digits


def swissroll():
    n_samples = 1500
    noise = 0.05
    X, _ = make_swiss_roll(n_samples, noise=noise)
    # Make it thinner
    X[:, 1] *= .5
    
    ward = AgglomerativeClustering(n_clusters=2, linkage='ward').fit(X)
    label = ward.labels_
    
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(7, -80)
    for l in np.unique(label):
        ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
               color=plt.cm.jet(np.float(l) / np.max(label + 1)),
               s=20, edgecolor='k')
    plt.show()
    
def swissroll_data(labeledsplit = 0.3):
    n_samples = 1500
    noise = 0.05
    X, _ = make_swiss_roll(n_samples, noise=noise)
    # Make it thinner
    X[:, 1] *= .5
    
    ward = AgglomerativeClustering(n_clusters=2, linkage='ward').fit(X)
    label = ward.labels_
    
    nrows = X.shape[0]
    labeledrows = ceil(labeledsplit * (nrows - 1))
    
    y = label
    
    trainXL = X[0:labeledrows, :]
    trainYL = y[0:labeledrows]
    trainXU = X[labeledrows:, :]
    trainYU = y[labeledrows:]
    
    return trainXL, trainYL, trainXU, trainYU
    
def digits():
    digits, labels = load_digits(return_X_y = True)
    
def digits_data(labeledsplit = 0.3):
    digits, labels = load_digits(return_X_y = True)
    
    X = digits
    y = labels
    
    nrows = X.shape[0]
    labeledrows = ceil(labeledsplit * (nrows - 1))
    
    trainXL = X[0:labeledrows, :]
    trainYL = y[0:labeledrows]
    trainXU = X[labeledrows:, :]
    trainYU = y[labeledrows:]
    
    return trainXL, trainYL, trainXU, trainYU
    
def moons():
    n_samples = 1500
    # noisy_moons = datasets.make_moons(n_samples = n_samples, noise=.05)
    X, y = datasets.make_moons(n_samples = n_samples, noise=.05)
    X = StandardScaler().fit_transform(X)
    
    algo = cluster.AgglomerativeClustering(n_clusters = 2, linkage='single')
    
    algo.fit(X)
    y = algo.labels_.astype(np.int)
    
    plt.figure(figsize=(9 * 1.3 + 2, 14.5))
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                          int(max(y) + 1))))
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y])

    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())
    
    plt.show()
    
def moons_data(labeledsplit = 0.3):
    n_samples = 1500
    X, y = datasets.make_moons(n_samples = n_samples, noise=.05)
    X = StandardScaler().fit_transform(X)
    
    algo = cluster.AgglomerativeClustering(n_clusters = 2, linkage='single')
    
    algo.fit(X)
    y = algo.labels_.astype(np.int)
    
    nrows = X.shape[0]
    labeledrows = ceil(labeledsplit * (nrows - 1))
    
    trainXL = X[0:labeledrows, :]
    trainYL = y[0:labeledrows]
    trainXU = X[labeledrows:, :]
    trainYU = y[labeledrows:]
    
    return trainXL, trainYL, trainXU, trainYU
    
if __name__ == "__main__":
    moons()
