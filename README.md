## Customer Segmentation Analysis

Project Objective:

This project aims to segment customers into distinct groups based on their purchasing behavior using unsupervised learning techniques: K-Means and Hierarchical Clustering. By analyzing these segments, we can gain valuable insights and develop targeted marketing strategies.
# Libraries Used:

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- sklearn
- datetime
- yellowbrick
- scipy
- scikit-learn

# Methodology:
## 1. Data Loading and Preprocessing:

- Load the dataset using pandas.
- Check for missing values and handle them appropriately (e.g., imputation or removal).
- Convert date columns to datetime format.
- Create new features:
   - recency: Time elapsed since the last purchase.
   - tenure: Customer age (difference between first and last purchase dates).
- Explore the data visually using descriptive statistics and visualizations.


## 2. Data Preparation: 
- Select relevant numerical features for modeling.
- Address skewness in the data using log transformation (if necessary).
- Standardize the features using MinMaxScaler for better performance in clustering algorithms.

## 3. K-Means Clustering:

- Determine the optimal number of clusters using the Elbow Method with the KElbowVisualizer from Yellowbrick.
- Create a K-Means model with the chosen number of clusters and fit it to the scaled data.
- Segment customers based on their K-Means cluster assignments.
- Analyze the characteristics of each segment using descriptive statistics.
## 4. Hierarchical Clustering:

- Utilize the standardized data to create a dendrogram and visually identify the optimal number of clusters.
- Create a Hierarchical Clustering model (AgglomerativeClustering) with the



