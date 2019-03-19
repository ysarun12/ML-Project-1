# ML-Project-1

# Background

This project is about a mobile marketplace app with about 6000 sellers and millions of customers. Each seller has a number of categories of products with several brands and different items in each brand. 
The data includes total number of listings of products, brands, number of orders placed from that seller and the total value of the products sold from that seller. 
Each seller has customers following with numbers ranging from 10's to 1000's. Some sellers have shares in the marketplace app and some don't.

Our aim in the project is to identify if a seller responds to a new feature introduced on the Marketplace App. 

### We have used True Positve rate (TPR) as the metric to judge a model.

To achieve this we have implemented following techniques/algorithms:

## Exploratory Data Analysis
- Data frame is sorted in decreasing order of days since the seller joined, so we allotted nearest value of days_since_joined for missing values
- For the attributes current_listings_shown, current_brands_shown, current_categories_shown, total_follows, total_int_shares, total_ext_shares
  we dropped the null values as they are only 3.7% of the remaining dataset
- We found that users who responded to the new feature is about 75% of those who didn't respond.
- Newr users are more prone to respond the feature.
- Users with more internal share are more likely to respond to the new feature.


## Classification Algorithms:

### Support Vector Machines (SVM)
- Experimented with linear, polynomial and radial kernels and found that linear kernel has the least error rate.
- Achieved a TPR of 86%

### Decision Trees
- Used learning curves to prune and decide optimal depth of the tree to avoid overfitting. Fixed on a tree depth of 2 layers.
- Achieved a TPR of 74%

### Boosting Techniques (AdaBoost)
- Learning rate of 1 provided the best results (least error rate)
- Achieved a TPR of 80%

- Adaboost had high bias and Decision Trees had high variance. Using learning curves, we achieved a bias-variance trade-off using k fold cross validation in SVM.

### Artificial Neural Networks
- Used Learning curves to experiment with layer sizes, nodes and activation functions. The best model was with Tanh activation function and 3 layers of 10, 5 and 5 nodes each.
- Achieved a TPR of 81%

### K Nearest Neighbors
- Used Euclidean distance to define nearest neighbors. Used Learning curves and determined k(=10) fold CV gave good bias-variance trade-off.
- Achieved a TPR of 87%

## Clustering Algorithms:

### K-Means
- Using elbow curve determined optimal k=8. With 8 clusters achieved Silhoutte score of 0.2

### Expectation Maximization (EM)
- Using elbow method and BIC score optimum number of components was determined to be 4

### Implemented Neural Networks On K-Means and EM clustering results as features and class labels as output
- Neural network with 4 layers with node sizes of 35, 25, 15 and 5 perform the best with sigmoid activation function on K-means clusters
- Gave TPR of 0% so we rejected this model.
- Neural network with 4 layers with node sizes of 35, 25, 15 and 5 perform the best with ReLU activation function on EM clusters
- Gave a TPR of 88.4%! Highest till now.

## Feature Engineering:

- Decision Tree feature selection algorithm: With tree depth of 10 we got only 1 feature to be important
- Principal Component Analysis: We got 4 principal components 
- Independant Component Analysis: We got 4 components in this too
- Randomized Projections: Using forward selection method we got 8 optimal components

## Implemented clustering algorithms, K-Means and EM on the selected features after dimensionality reduction to get new number of clusters

### K-Means				
| Feature Selection Method | DT | PCA |	ICA |	RP |
| ------------------------ | -- | --- | --- | -- |
| Number of Clusters	     | 10 |	12	| 10	| 10 |

### Expectation Maximization				
| Feature Selection Method | DT |	PCA |	ICA |	RP |
| ------------------------ | -- | --- | --- | -- |
| Number of Clusters	     | 11	|  5	|  5	| 4  |


## Implemented Neural Networks on the selected features after dimensionality reduction

|	Dimensionality Reduction | No. of Layers |	Activation Function |	 TPR   |
| ------------------------ | ------------- | -------------------- | ------ |    
| DT	                     |    3	         |      ReLU	          |  82.3% |
| PCA	                     |    4	         |      ReLU	          |  81.6% |
| ICA	                     |    4	         |      tanh	          |  83.3% |
| RP	                     |    4	         |      tanh	          |  86.5% |


# Results

- SVM performed better because our data is linearly seperable and hence suited for SVM
- Data represents consumer behavior, KNN finds users most similar to each other and hence suited for the given problem statement

### Comparison within Neural Networks Using only Clusters as Input
ANN model trained just on the clusters from k-means didn't perform well whereas the ANN trained on the clusters from EM did a great job in classifying 'True Positive' responses by the user. This makes sense as soft clustering would better describe the data given it's nature and hence the highest TPR of 89%!

### Comparison within Neural Networks Using only Reduced Dimensions as Input
All the ANN performed well after dimentionality reduction with true positive rate of over 82% 
The best performing one was randomized projections for this problem with a TPR of about 89%
