# Creating a clustering dataset, plotting it, and saving as an SVG.
# This code will:
# 1. Generate a 2D clustering dataset with sklearn.make_blobs
# 2. Show the first rows of the dataset in a table for inspection
# 3. Plot the clusters with matplotlib (single plot, no explicit color choices)
# 4. Save the plot as an SVG at /mnt/data/cluster_plot.svg
#
# After this cell runs, a link to download the SVG will be provided in the chat.

from sklearn.datasets import make_blobs
import pandas as pd
import matplotlib.pyplot as plt
import os

# generate dataset
X, y = make_blobs(n_samples=500, centers=4, cluster_std=[1.0, 2.5, 0.5, 1.2], random_state=42)

# create DataFrame for easy inspection
df = pd.DataFrame(X, columns=["x", "y"])
df["cluster"] = y

# display a preview to the user using the helper from the notebook environment
try:
    from caas_jupyter_tools import display_dataframe_to_user
    display_dataframe_to_user("Clustering dataset preview (first 10 rows)", df.head(10))
except Exception:
    # If helper not available, just print first rows
    print(df.head(10).to_string(index=False))

# Plot (single plot, using matplotlib; do not explicitly set colors)
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(df["x"], df["y"], c=df["cluster"], s=30, alpha=0.8)
# ax.set_title("Clustering dataset")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True, linestyle=':', linewidth=0.5)

# Save as SVG
out_path = "/home/1-profiling/imgs/cluster_plot"
plt.savefig(out_path + ".svg", bbox_inches="tight")
plt.savefig(out_path + ".pdf", bbox_inches="tight")
# plt.show()

print(f"\nSaved SVG to: {out_path}")



# Re-run after reset: Create embeddings, plot in 2D, and compute cosine similarities.
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Words to embed
words = ['telephone', 'first_name', 'customer_id', 'country', 'email', 'location', 'name']

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings
embeddings = model.encode(words)

print(embeddings)

# Reduce to 2D with PCA
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(embeddings)

# Plot
fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(embeddings_2d[:,0], embeddings_2d[:,1], s=50)

for i, word in enumerate(words):
    ax.text(embeddings_2d[i,0]+0.02, embeddings_2d[i,1]+0.02, word, fontsize=10)

ax.set_title("Word Embeddings (PCA to 2D)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.grid(True, linestyle=":")

# plt.show()
out_path = "/home/1-profiling/imgs/embeddings_2d"
plt.savefig(out_path + ".svg", bbox_inches="tight")
plt.savefig(out_path + ".pdf", bbox_inches="tight")

# Cosine similarity function
def cos_sim(word1, word2):
    i1, i2 = words.index(word1), words.index(word2)
    return cosine_similarity([embeddings[i1]], [embeddings[i2]])[0,0]

# Compute requested similarities
sims = {
    ("country", "location"): cos_sim("country", "location"),
    ("location", "name"): cos_sim("location", "name"),
    ("name", "first_name"): cos_sim("name", "first_name")
}

print(sims)
