"""K-means clustering for 1D input
        
Properties:
    k {int} -- Number of clusters

"""

import numpy as np

class KMeans:
    
    def __init__(self, k):
        self.k = k
        self.clusters = 0
        self.stds = np.zeros(self.k)
        
    def fit(self,X):
        """Performs k-means clustering for 1D input
        
        Arguments:
            X {ndarray} -- A Mx1 array of inputs
            k {int} -- Number of clusters
        
        Returns:
            ndarray -- A kx1 array of final cluster centers
        """
        # randomly select initial clusters from input data
        self.clusters = np.random.choice(np.squeeze(X), size=self.k)
        prevClusters = self.clusters.copy()
        
        converged = False
        while not converged:
            """
            compute distances for each cluster center to each point 
            where (distances[i, j] represents the distance between the ith point and jth cluster)
            """
            distances = np.squeeze(np.abs(X[:, np.newaxis] - self.clusters[np.newaxis, :]))
            # find the cluster that's closest to each point
            closestCluster = np.argmin(distances, axis=1)
            # update clusters by taking the mean of all of the points assigned to that cluster
            for i in range(self.k):
                pointsForCluster = X[closestCluster == i]
                if len(pointsForCluster) > 0:
                    self.clusters[i] = np.mean(pointsForCluster, axis=0)
            # converge if clusters haven't moved
            converged = np.linalg.norm(self.clusters - prevClusters) < 1e-6
            prevClusters = self.clusters.copy()
        distances = np.squeeze(np.abs(X[:, np.newaxis] - self.clusters[np.newaxis, :]))
        closestCluster = np.argmin(distances, axis=1)
        clustersWithNoPoints = []
        for i in range(self.k):
            pointsForCluster = X[closestCluster == i]
            if len(pointsForCluster) < 2:
                # keep track of clusters with no points or 1 point
                clustersWithNoPoints.append(i)
                continue
            else:
                self.stds[i] = np.std(X[closestCluster == i])
        # if there are clusters with 0 or 1 points, take the mean std of the other clusters
        if len(clustersWithNoPoints) > 0:
            pointsToAverage = []
            for i in range(self.k):
                if i not in clustersWithNoPoints:
                    pointsToAverage.append(X[closestCluster == i])
            pointsToAverage = np.concatenate(pointsToAverage).ravel()
            self.stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))