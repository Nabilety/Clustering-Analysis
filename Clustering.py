import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# # Grouping objects by similarity using k-means

# ## K-means clustering using scikit-learn

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

plt.scatter(X[:, 0],
            X[:, 1],
            c='white',
            marker='o',
            edgecolor='black',
            s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
plt.tight_layout()
plt.show()

# Goal is to group examples based on their feature similarities, achieved using k-means algorithm with the steps:
# 1. Randomly pick k centroids from the examples as initial cluster centers.
# 2. Assign each example to the nearest centroid
# 3. Move the centroids to the center of the examples that were assigned to it
# 4. Repeat step 2 and 3 until the cluster assignments do not change or a user-defined tolerance or maximum number of iterations is reach

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

# note n_init=10 will run the k-means clustering 10 times independently, with different centroids and choose final model
# as the one with the lowest Sum of Squared errors (SSE). Max_iter specifies maximum number of iterations for each single run
# note that k-means in scikit-learn stops early if it converges before maximum number of iterations is reached.
# But if it doesn't converge it can be problematic (computationally) if we choose relatively large values for max_iter.
# one way to deal with convergence problem is to choose larger values for tol, that controls tolerance regarding changes
# in the within-cluster SSE to declare convergence. In this example we chose tolerance 1e-04 (=0.0001)

# clusters can be empty, but this is accounted for in scikit-learn
# as it will search for the example farthest away from the centroid of the empty cluser.
# Then it will reassign the centroid to bte hte farthest point

# Visualize cluster labels and cluster centroids (cluster_centers_ attr) predicted by k-means:
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='Cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='Cluster 2')
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50, c='lightblue',
            marker='v', edgecolor='black',
            label='Cluster 3')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()
# while the k-means worked well on this toy dataset, the drawback is having to specify number of k clusters, a priori
# and this may not always be obvious in real-world applications, especially working with higher-dimensional dataset.
# Aside from that property k-means clusters don't overlap, are not hierarchichal and we also assume at least one item in each cluster

# ## A smarter way of placing the initial cluster centroids using k-means++

# ...

# ## Hard versus soft clustering

# ...

# Within-cluster SSE using scikit-learn attribute after iftting KMeans model
print(f'Distortion: {km.inertia_:.2f}')

# ## Using the elbow method to find the optimal number of clusters

distorions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distorions.append(km.inertia_)
plt.plot(range(1, 11), distorions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()

# ## Quantifying the quality of clustering via silhouette plots
# Create plot of silhouette coefficients for a k-means clustering with k=3
km = KMeans(n_clusters=3,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e+04,
            random_state=0)
y_km = km.fit_predict(X)

from matplotlib import cm
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
            color='red',
            linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
plt.show()

# through visual insepction of the silhouette plot we can quickly scrutinize the sizes of the different clusters and
# and identify clusters that contain outliers: however as we can see in this plot, the silhouette coefficients are not
# close to 0 and are approximately equally far away from the average silhouette score, which is in this case an indicator
# of good clustering. Furthermore to summarize the goodness of our clustering we added the average silhouette coefficient to plot

# to see what a silhouette plot looks like for a relatively bad clustering, let's seed the k-means algorithm with only two centroids

# Comparison to "bad" clustering:
km = KMeans(n_clusters=2,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            edgecolor='black',
            marker='s',
            label='Cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='orange',
            edgecolor='black',
            marker='o',
            label='Cluster 2')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='red',
            label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# keep in mind we typically do not have the luxury of visualizing datasets in two-dimensional scatterplots in real-world
# problems, since we typically work with data in higher dimensions. So next we create silhouette plot to evaluate the results
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
             edgecolor='none', color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
# plt.savefig('figures/10_06.png', dpi=300)
plt.show()

# the silhouettes now have visibly different lengths and widths, which is
# evidence of a relatively bad or at least suboptimal clustering:


# # Organizing clusters as a hierarchical tree

# ## Grouping clusters in bottom-up fashion

# Hierarchical complete linkage clustering:
# 1. Compute a pair-wise distance matrix of all examples
# 2. Represent each data point as a singleton cluster
# 3. Merge the two closest clusters based on the distance between the most dissimilar (distant) members
# 4. Update the cluster linkage matrix
# 5. Repeat steps 2-4 until one single cluster remains

# Generate random data sample - rows represent different observations (IDs 0-4) and columns different features (X, Y, Z)
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)

# ## Performing hierarchical clustering on a distance matrix
from scipy.spatial.distance import pdist, squareform
row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),
                        columns=labels, index=labels)
print(row_dist)
# preceding code calculated the euclidean distance between each pair of input examples, based on the features X, Y, Z
# pdist returned a condensed distance matrix, as input to the squareform function, to create a symmetrical matrix of
# pair-wise distances

# apply complete linkage agglomeration to our clusters using linkage function from scipy's submodule, returning a linkage matrix
# first check the module docs
from scipy.cluster.hierarchy import linkage
#help(linkage)

# We can either pass a condensed distance matrix (upper triangular) from the `pdist` function,
# or we can pass the "original" data array and define the `metric='euclidean'` argument in `linkage`.
# However, we should not pass the squareform distance matrix, which would yield different distance values
# although the overall clustering could be the same.



# 1. incorrect approach: Squareform distance matrix
row_clusters = linkage(row_dist, method='complete', metric='euclidean')
pd.DataFrame(row_clusters,
             columns=['row label 1', 'row label 2',
                      'distance', 'no. of items in clust.'],
             index=[f'cluster {(i + 1)}'
                    for i in range(row_clusters.shape[0])])


# 2. correct approach: Condensed distance matrix
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
pd.DataFrame(row_clusters,
             columns=['row label 1', 'row label 2',
                      'distance', 'no. of items in clust.'],
            index=[f'cluster {(i + 1)}'
                    for i in range(row_clusters.shape[0])])


# 3. correct approach: Input matrix
row_clusters = linkage(df.values, method='complete', metric='euclidean')
pd.DataFrame(row_clusters,
             columns=['row label 1', 'row label 2',
                      'distance', 'no. of items in clust.'],
             index=[f'cluster {(i + 1)}'
                    for i in range(row_clusters.shape[0])])


from scipy.cluster.hierarchy import dendrogram
# make dendrogram black (part 1/2)
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(['black'])
row_dendr = dendrogram(row_clusters,
                       labels=labels,
                       # make dendrogram black (part 2/2)
                       # color_threshold=np.inf
                       )
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()


# ## Attaching dendrograms to a heat map
# Attaching a dendrogram to a heat map can be tricky so lets go through it step by step:

# 1: Create new figure object and define the x-axis and y-axis positions, width, height of the dendrogram via the
# add_axes attribute. Furthermore rotate the dendrogram 90 degrees counterclockwise
fig = plt.figure(figsize=(8, 8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters,
                       orientation='left')
# note: for matplotlib < v1.5.1, please use orientation='right'

# 2. Reorder data in our initial DF according to the clustering labels that can be accessed from dendrogram object
# which essentially is a python dictionary via the leaves key
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]

# 3. Construct heat map from the reordered DF and position it next to the dendrogram
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust,
                  interpolation='nearest',
                  cmap='hot_r')

# 4. Modify aesthethics of the dendrogram by removing the axis ticks and hiding the axis spines.
# also add a color bar and assign the feature and data record names to the x and y axis tick labels respectively.
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()


# ## Applying agglomerative clustering via scikit-learn
# AgglomerativeClustering allows us to choose number of clusters to return, which is useful if we want to prune the hierarchical cluster tree
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3,
                             affinity='euclidean',
                             linkage='complete')
labels = ac.fit_predict(X)
print(f'Cluster labels: {labels}')
# Cluster labels: [1 0 0 2 1]
# Looking at the predicted cluster labels, we see that the first and fifth examples (ID_0 and ID_4)
# we assigned to one cluster (label 1), and examples ID_1 and ID_2 assigned to a second cluster (label 0)
# Exampel ID_3 was put into its own cluster (label 2). This is consistent with the results we observed in the dendrogram
# Note that ID_3 was more similar to ID_4 and ID_0 than ID_1 and ID_2 as shown in the dendrogram. This is not clear in
# Scikit-learns clustering result

# Rerun using n_clusters=2
ac = AgglomerativeClustering(n_clusters=2,
                             affinity='euclidean',
                             linkage='complete')
labels = ac.fit_predict(X)
print(f'Cluster labels: {labels}')
# Cluster labels: [0 1 1 0 0]
# As we can see, in this pruned clustering hierarchy, label ID_3 was assigned to the same cluster as ID_0
# and ID_4, as expected




# # Locating regions of high density via DBSCAN
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200,
                  noise=0.05,
                  random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.tight_layout()
plt.show()

# K-means and hierarchical clustering:
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X)
ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
            edgecolor='black',
            c='lightblue', marker='o', s=40, label='cluster 1')
ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
            edgecolor='black',
            c='red', marker='s', s=40, label='cluster 2')
ax1.set_title('K-means clustering')

ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

ac = AgglomerativeClustering(n_clusters=2,
                             affinity='euclidean',
                             linkage='complete')
y_ac = ac.fit_predict(X)
ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c='lightblue',
            edgecolor='black',
            marker='o', s=40, label='Cluster 1')
ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], c='red',
            edgecolor='black',
            marker='s', s=40, label='Cluster 2')
ax2.set_title('Agglomerative clustering')

ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')

plt.legend()
plt.tight_layout()
plt.show()

# Neither of the two algorithms are capable of separating the two moon-shaped clusters

# Density-based clustering:
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2,
            min_samples=5,
            metric='euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db == 0, 0],
            X[y_db == 0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o',
            s=40,
            label='Cluster 1')
plt.scatter(X[y_db == 1, 0],
            X[y_db == 1, 1],
            c='red',
            edgecolor='black',
            marker='s',
            s=40,
            label='Cluster 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.tight_layout()
plt.show()

# A shown, the DBSCAN can successfully detect the half-moon shapes, which is one of its strengths -
# clustering data of arbitrary shapes. However some of the disadvantages of DBSCAN includes:
# Assuming a fixed number of examples, the curse of dimensionality increase with increasing number of features in our set
# This especially occurs when using Euclidean distance metric, but is not unique for this algorithm as it also appears
# in k-means and hierarchical clusterings.
# Additionally we have two hyperparameters (MinPts and epsilon) that needs optimization to yield best results.
# In practice applying dimensionality reduction techniques prior to performing clustering is common, i.e. t-SNE, component analysis
# similarly it is common to compress datasets down to two-dimensional subspaces, which allows us to visualize clusters and assigned
# labels using two-dimensional scatterplots that helps us evaluate the results

