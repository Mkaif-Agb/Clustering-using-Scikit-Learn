# Clustering-using-Scikit-Learn
In general, there are two types of machine learning algorithms, Supervised Machine Learning and Unsupervised Machine Learning. In addition, new categories evolve with development in the field which can be identified as reinforcement learning. Let’s dive into what these categories are and how they work.
## Supervised Learning
In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output. Supervised learning problems are further categorized into regression and classification problems.
## Unsupervised Learning
On the contrary to Supervised learning, Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don’t necessarily know the effect of the variables.
We can derive this structure by clustering the data based on relationships among the variables in the data. With Unsupervised learning there is no feedback based on the prediction results. For example, take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on. This is a good example of clustering. Whereas, for a non-clustering problem such as “Cocktail Party Problem”, it helps in identifying voices music from a mesh of sounds at a cocktail party.
# K-means (Clustering)
![K-Means Example](https://www.mathworks.com/help/examples/stats/win64/PartitionDataIntoTwoClustersExample_02.png)

K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity. The results of the K-means clustering algorithm are:
The centroids of the K clusters, which can be used to label new data
Labels for the training data (each data point is assigned to a single cluster)
Rather than defining groups before looking at the data, clustering allows you to find and analyze the groups that have formed organically. The "Choosing K" section below describes how the number of groups can be determined.(We did this using the elbow method with wcss)
![K-Means Cluster for Dataset Provided](https://github.com/Mkaif-Agb/Clustering-using-Scikit-Learn/blob/master/KMeansClustering.png?raw=true)
![Elbow Method](https://miro.medium.com/proxy/0*jWe7Ns_ubBpOaemM.png)
![Elbow Method for Dataset Provided](https://github.com/Mkaif-Agb/Clustering-using-Scikit-Learn/blob/master/Elbow%20Method.png?raw=true)

# Hierarchical Clustering

![Hierarchical Example](http://dataaspirant.com/wp-content/uploads/2018/01/Hierarchical_clustering_agglomerative_and_divisive_methods.png)

Hierarchical clustering arranges items in a hierarchy with a treelike structure based on the distance or similarity between them. The graphical representation of the resulting hierarchy is a tree-structured graph called a dendrogram. In Spotfire, hierarchical clustering and dendrograms are strongly connected to heat map visualizations. You can cluster both rows and columns in the heat map. Row dendrograms show the distance or similarity between rows, and which nodes each row belongs to as a result of clustering. Column dendrograms show the distance or similarity between the variables (the selected cell value columns). The example below shows a heat map with a row dendrogram.
# Agglomerative Clustering: 
Also known as bottom-up approach or hierarchical agglomerative clustering (HAC). A structure that is more informative than the unstructured set of clusters returned by flat clustering. This clustering algorithm does not require us to prespecify the number of clusters. Bottom-up algorithms treat each data as a singleton cluster at the outset and then successively agglomerates pairs of clusters until all clusters have been merged into a single cluster that contains all data.
# Divisive clustering : 
Also known as top-down approach. This algorithm also does not require to prespecify the number of clusters. Top-down clustering requires a method for splitting a cluster that contains the whole data and proceeds by splitting clusters recursively until individual data have been splitted into singleton cluster.
![Hierarchical Cluster for Dataset provided](https://github.com/Mkaif-Agb/Clustering-using-Scikit-Learn/blob/master/Heirarchichal_Clusters.png?raw=true)
![Dendogram to Know the number of clusters](https://www.saedsayad.com/images/Clustering_h1.png)
![Dendogram for Dataset provided](https://github.com/Mkaif-Agb/Clustering-using-Scikit-Learn/blob/master/Dendrogram_Plot.png?raw=true)
