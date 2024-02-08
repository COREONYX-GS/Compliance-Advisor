import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

print(os.getcwd())

def Step1():
    # Load the Excel file for initial exploration
    file_path = 'sp800-171r2-security-reqs.xlsx'
    nist_800_171_df = pd.read_excel(file_path, sheet_name='SP 800-171')

    # Extract the security requirements and discussions for analysis
    requirements = nist_800_171_df['Family'].tolist()
    discussions = nist_800_171_df['Discussion'].tolist()

    # Combine requirements and discussions for a more comprehensive analysis
    combined_texts = [req + " " + disc for req, disc in zip(requirements, discussions)]

    # Step 1: Vectorize the text using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_texts)

    # Displaying the shape of the TF-IDF matrix to understand its structure
    print(tfidf_matrix.shape)

    return nist_800_171_df, tfidf_matrix

def Step2(tfidf_matrix):
    # Compute the cosine similarity matrix
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Since the matrix is quite large, let's display a small portion to understand its structure
    print(cosine_similarities[:5, :5])

    return cosine_similarities

def Step3(cosine_similarities, nist_800_171_df):
    # Reduce dimensions for visualization, aiming for a 2D representation
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(1 - cosine_similarities)  # 1 - cosine_similarities to convert to distance matrix

    # Plotting the node diagram
    plt.figure(figsize=(12, 8))
    plt.scatter(pos[:, 0], pos[:, 1], marker='o')
    for i, txt in enumerate(nist_800_171_df['Identifier'].tolist()):
        plt.annotate(txt, (pos[i, 0], pos[i, 1]))
    plt.title("Node Diagram of NIST SP 800-171 Controls")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.grid(True)
    plt.show()

def BAD_plot_with_clusters(data, n_neighbors=5):
    """
    Plot the data with ellipses around clusters.
    `n_neighbors` determines the size of the clusters.
    """

    # Reduce dimensions for visualization, aiming for a 2D representation
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(1 - cosine_similarities)  # 1 - cosine_similarities to convert to distance matrix

    # Using NearestNeighbors to find the clusters
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    distances, indices = nbrs.kneighbors(data)

    # Plotting the data
    plt.figure(figsize=(12, 8))
    plt.scatter(data[:, 0], data[:, 1], marker='o')

    # Adding ellipses to the plot
    for i in range(data.shape[0]):
        mean = np.mean(data[indices[i]], axis=0)
        cov = np.cov(data[indices[i]].T)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, color='red', alpha=0.2)
        plt.gca().add_patch(ellipse)

    # Labeling the plot
    for i, txt in enumerate(data['Identifier'].tolist()):
        plt.annotate(txt, (data[i, 0], data[i, 1]))
    plt.title("NIST SP 800-171 Controls with Clustering")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.grid(True)
    plt.show()

    # Plotting the data with clusters
    plot_with_clusters(pos, n_neighbors=5)  # Feel free to modify `n_neighbors` for different cluster sizes

def Step4(nist_df, tfidf_matrix, cosine_similarities):

    # Load the Excel file for initial exploration
    file_path = 'sp800-171r2-security-reqs.xlsx'
    
    nist_800_171_df = pd.read_excel(file_path, sheet_name='SP 800-171')

    # Extract the security requirements and discussions for analysis
    requirements = nist_800_171_df['Family'].tolist()
    discussions = nist_800_171_df['Discussion'].tolist()

    # Combine requirements and discussions for a more comprehensive analysis
    combined_texts = [req + " " + disc for req, disc in zip(requirements, discussions)]

    # Vectorize the text using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_texts)

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Reduce dimensions for visualization, aiming for a 2D representation
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(1 - cosine_similarities)  # 1 - cosine_similarities to convert to distance matrix

    def plot_with_clusters(nist_df, data, n_neighbors=5):
        """
        Plot the data with ellipses around clusters.
        `n_neighbors` determines the size of the clusters.
        """
        # Using NearestNeighbors to find the clusters
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
        distances, indices = nbrs.kneighbors(data)

        # Plotting the data
        plt.figure(figsize=(12, 8))
        plt.scatter(data[:, 0], data[:, 1], marker='o')

        # Adding ellipses to the plot
        for i in range(data.shape[0]):
            mean = np.mean(data[indices[i]], axis=0)
            cov = np.cov(data[indices[i]].T)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(vals)
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, color='red', alpha=0.2)
            plt.gca().add_patch(ellipse)

        # Labeling the plot
        for i, txt in enumerate(nist_df['Identifier'].tolist()):
            plt.annotate(txt, (data[i, 0], data[i, 1]))
        plt.title("NIST SP 800-171 Controls with Clustering")
        plt.xlabel("MDS Dimension 1")
        plt.ylabel("MDS Dimension 2")
        plt.grid(True)
        plt.show()

    # Plotting the data with clusters
    plot_with_clusters(nist_df, pos, n_neighbors=5)  # Feel free to modify `n_neighbors` for different cluster sizes

def Step5(cluster_size):
    # Load the Excel file for initial exploration
    file_path = 'sp800-171r2-security-reqs.xlsx'
    
    nist_800_171_df = pd.read_excel(file_path, sheet_name='SP 800-171')

    # Extract the security requirements and discussions for analysis
    requirements = nist_800_171_df['Family'].tolist()
    discussions = nist_800_171_df['Discussion'].tolist()

    # Combine requirements and discussions for a more comprehensive analysis
    combined_texts = [req + " " + disc for req, disc in zip(requirements, discussions)]

    # Vectorize the text using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_texts)

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Reduce dimensions for visualization, aiming for a 2D representation
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(1 - cosine_similarities)  # 1 - cosine_similarities to convert to distance matrix

    def plot_with_clusters(nist_df, data, n_neighbors):
        """
        Plot the data with ellipses around clusters.
        `n_neighbors` determines the size of the clusters.
        """
        # Using NearestNeighbors to find the clusters
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
        distances, indices = nbrs.kneighbors(data)

        # Plotting the data
        plt.figure(figsize=(12, 8))
        plt.scatter(data[:, 0], data[:, 1], marker='o')

        # Adding ellipses to the plot
        for i in range(data.shape[0]):
            mean = np.mean(data[indices[i]], axis=0)
            cov = np.cov(data[indices[i]].T)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(vals)
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, color='red', alpha=0.2)
            plt.gca().add_patch(ellipse)

        # Labeling the plot
        for i, txt in enumerate(nist_df['Identifier'].tolist()):
            plt.annotate(txt, (data[i, 0], data[i, 1]))
        plt.title("NIST SP 800-171 Controls with Clustering")
        plt.xlabel("MDS Dimension 1")
        plt.ylabel("MDS Dimension 2")
        plt.grid(True)
        plt.show()

    # Plotting the data with clusters
    plot_with_clusters(nist_df, pos, cluster_size)

if __name__ == "__main__":
    nist_df, tfidf_matrix = Step1()
    cosine_similarities = Step2(tfidf_matrix)
  
    #Step3(cosine_similarities, nist_df)
  
    #plot_with_clusters(nist_df, n_neighbors=5)

    #Step4(nist_df, tfidf_matrix, cosine_similarities)

    for n in range(1, 10):
        print("Graph for: ", n)
        Step5(n)
      
    print("Done")
