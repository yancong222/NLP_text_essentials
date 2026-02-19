# import libraries
import numpy as np
import pandas as pd
import re
import string
import os
import math
import csv
import shutil, sys

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

"""# Data Preprocessing"""

# load the data
df = pd.read_csv(data + 'data.csv')
df.head()

"""clean the "text": remove emoji, URLs, hashtags"""

# Function to clean text

import re
import pandas as pd

def clean_text(text):
    # Check if the input is a string
    if isinstance(text, str):

      #Convert text to lowercase
        text = text.lower()

        # Remove new lines and extra spaces
        text = text.replace('\n', ' ').strip()

        # Remove hyperlinks (URLs)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove hashtags
        text = re.sub(r'#\w+', '', text)

        # Remove emojis (basic pattern to match most common emojis)
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F700-\U0001F77F"  # alchemical symbols
                                   u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                   u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                   u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                   u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                   u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                   u"\U00002702-\U000027B0"  # Dingbats
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        return text
    else:
        # Handle non-string values (e.g., return empty string or original value)
        return str(text) # or return '' or return text

# Apply the cleaning function to the 'text' column
df['clean_text'] = df['text'].apply(clean_text)

df[['id', 'text',"clean_text"]].head()

"""# K-means Clustering (Unsupervised)

Clustering will allow us to group the posts into different clusters without any labels.

K-Means is an unsupervised learning algorithm that will help us group similar posts together based on the content of their cleaned_text.

After clustering, we can analyze the groups to identify which clusters contain relevant or irrelevant posts.

## Step 1: Vectorize the Text Data
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Convert text to numerical features using TF-IDF
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
  # max_features=1000, It tells the vectorizer to keep only the top 500 most important words, based on their TF-IDF scores.

X_tfidf = tfidf.fit_transform(df['clean_text'])

# Get the list of words (terms) from the TF-IDF vectorizer
feature_names = tfidf.get_feature_names_out()

# Convert the TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=feature_names)

# Display the first 5 rows of the DataFrame
print(tfidf_df.head())

"""## Step 2: find the right number of cluster

**How Can We Know How Many Clusters We Need?**

Determining the optimal number of clusters is a key challenge in clustering algorithms like K-Means. There are several methods to help identify the best number of clusters:

**1) Elbow Method**

The Elbow Method involves running K-Means for a range of cluster numbers (e.g., from 1 to 10) and plotting the within-cluster sum of squared distances (WCSS), also known as inertia.

The idea is to find the "elbow" point where adding more clusters does not significantly improve the fit.
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Calculate WCSS (inertia) for different numbers of clusters
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_tfidf)
    wcss.append(kmeans.inertia_)

# Step 2: Plot the Elbow Curve
plt.plot(range(1, 20), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Inertia)')
plt.show()

"""**2) Silhouette Score**

Another method is to use the Silhouette Score, which measures how similar each point is to its own cluster compared to other clusters. A higher silhouette score indicates better-defined clusters.
"""

from sklearn.metrics import silhouette_score

# Step 1: Compute silhouette scores for different numbers of clusters
silhouette_scores = []
for i in range(2, 11):  # Silhouette score requires at least 2 clusters
    kmeans = KMeans(n_clusters=i, random_state=42)
    cluster_labels = kmeans.fit_predict(X_tfidf)
    score = silhouette_score(X_tfidf, cluster_labels)
    silhouette_scores.append(score)

# Step 2: Plot the Silhouette Score
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

"""## Step 3: Apply K-Means Clustering

We'll then apply K-Means clustering to the TF-IDF matrix to group the posts into clusters.
"""

#Apply K-Means clustering
num_clusters = 9  # You can adjust the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_tfidf)

# Inspect the clusters
print(df[['clean_text', 'cluster']].head())

"""## Step 4 visualize the clusters

- t-SNE reduces the high-dimensional TF-IDF vectors to two dimensions for visualization.
- Each point represents a post, and the color corresponds to the cluster it belongs to.
- The scatter plot will give you an idea of how the posts are grouped into clusters.
"""

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Apply t-SNE to reduce the dimensionality to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_2d = tsne.fit_transform(X_tfidf.toarray())  # Convert sparse matrix to dense before t-SNE

# Step 2: Create a scatter plot with distinct colors for each cluster
plt.figure(figsize=(10, 6))

# Use a qualitative colormap for discrete colors (e.g., 'tab10' or 'Set1')
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['cluster'], cmap='tab10', marker='o', s=50, alpha=0.7)

# Add titles and labels
plt.title('t-SNE Visualization of Clusters with Discrete Colors')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

# Remove continuous color bar
plt.legend(*scatter.legend_elements(), title="Cluster")

# Show the plot
plt.show()

# Count the number of posts in each cluster
cluster_counts = df['cluster'].value_counts()

# Display the number of posts in each cluster
print(cluster_counts)

# Plot the number of posts in each cluster
plt.figure(figsize=(10, 6))
cluster_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Posts in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Posts')
plt.xticks(rotation=0)
plt.show()

"""## Step 5: Analyze the Clusters

After clustering, each post will be assigned a cluster label (stored in the cluster column). You can inspect the posts in each cluster to see which ones might correspond to irrelevant or promotional content.

Identify irrelevant or promotional content: Look for clusters that contain keywords or posts that look like advertisements or promotional content.
Group similar posts: See if clusters are grouping posts around specific topics (e.g., discussions about a particular issue).
Refine your approach: If certain clusters seem mixed (containing both relevant and irrelevant content), you might adjust the clustering or use additional filtering.

**(1) Extract Top Words for Each Cluster**

We’ll first extract top 10 words for each cluster using the TF-IDF matrix. These top words will give us a general idea of what each cluster is about.
"""

# Get the feature names (words) from the TF-IDF vectorizer
terms = tfidf.get_feature_names_out()

# Print the top 10 words for each cluster
for i in range(num_clusters):
    print(f"Cluster {i}:")
    # Get the indices of the top terms in the cluster center (centroid)
    top_terms_indices = np.argsort(kmeans.cluster_centers_[i])[-15:]  # Top 10 terms
    top_terms = [terms[ind] for ind in top_terms_indices]
    print(", ".join(top_terms))
    print("\n")

"""**(2) Inspect Posts in Each Cluster**

Once we’ve looked at the top words, we can further inspect the actual content of the posts in each cluster to better understand the theme of each group.

"""

import numpy as np

# Get the feature names (words) from the TF-IDF vectorizer
terms = tfidf.get_feature_names_out()

# Inspect a few posts from each cluster along with top 10 words
for i in range(num_clusters):
    print(f"Cluster {i} posts:")

    # Get the top 10 words for this cluster
    top_terms_indices = np.argsort(kmeans.cluster_centers_[i])[-10:]  # Top 10 terms
    top_terms = [terms[ind] for ind in top_terms_indices]

    # Print the top 10 words
    print(f"Top words: {', '.join(top_terms)}")

    # Get the sample posts from the cluster
    sample_posts = df[df['cluster'] == i]['clean_text'].head(5)  # Get first 5 posts from cluster i
    for post in sample_posts:
        print(f"- {post}")

    print("\n")  # Newline for readability between clusters

"""**if you want to see the posts in Cluster 0, you can do this:**"""

# Filter posts in cluster 0
cluster_0_posts = df[df['cluster'] == 0]

cluster_0_posts[['clean_text']].head(10)

"""## Step6 delete rows with specific clusters"""

# Count the number of unique types in the 'cluster' column
num_unique_clusters = df['cluster'].nunique()
print("Number of unique clusters:", num_unique_clusters)

# Display the unique values in the 'cluster' column
unique_clusters = df['cluster'].unique()
print("Unique clusters:", unique_clusters)

# Drop rows where 'cluster' is in [2, 4, 6]
df.drop(df[df['cluster'].isin([2, 4, 6])].index, inplace=True)

# Count the number of unique types in the 'cluster' column
num_unique_clusters = df['cluster'].nunique()
print("Number of unique clusters:", num_unique_clusters)

# Display the unique values in the 'cluster' column
unique_clusters = df['cluster'].unique()
print("Unique clusters:", unique_clusters)

# Save the rusult
df.to_csv(result + 'data.csv')

