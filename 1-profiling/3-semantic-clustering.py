import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# Helper functions
# -------------------------------
def extract_schema(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            return set(pd.read_csv(file_path, nrows=5).columns)
        elif ext in [".xlsx", ".xls"]:
            xl = pd.ExcelFile(file_path)
            keys = set()
            for sheet in xl.sheet_names:
                keys.update(xl.parse(sheet, nrows=5).columns)
            return keys
        elif ext == ".json":
            with open(file_path, "r") as f:
                data = json.load(f)
            keys = set()
            if isinstance(data, list):
                for item in data[:5]:
                    if isinstance(item, dict):
                        keys.update(item.keys())
            elif isinstance(data, dict):
                keys.update(data.keys())
            return keys
        else:
            return set()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return set()

def compute_column_embeddings(all_columns, model):
    return {col: model.encode(col, convert_to_tensor=True, normalize_embeddings=True) for col in all_columns}

def semantic_jaccard_similarity(schema_a, schema_b, col_embeddings, sim_threshold=0.7):
    if not schema_a or not schema_b:
        return 0.0
    match_count = 0
    for col_a in schema_a:
        sims = [util.cos_sim(col_embeddings[col_a], col_embeddings[col_b]).item() for col_b in schema_b]
        if max(sims) >= sim_threshold:
            match_count += 2 if col_a not in schema_b else 1
    union_count = len(schema_a.union(schema_b))
    return match_count / union_count if union_count > 0 else 0.0

# -------------------------------
# Main clustering function
# -------------------------------
def semantic_file_clustering_jaccard(base_path, sim_threshold=0.7, distance_threshold=0.3, output_file="semantic_file_clusters.json"):
    # Step 1: extract schemas
    file_schemas = {}
    all_columns = set()
    for root, _, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            schema = extract_schema(file_path)
            if schema:
                file_schemas[file_path] = schema
                all_columns.update(schema)
    
    if not file_schemas:
        print("No schemas found")
        return {}

    # Step 2: compute embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    col_embeddings = compute_column_embeddings(all_columns, model)

    # Step 3: compute semantic Jaccard similarity matrix
    files = list(file_schemas.keys())
    n = len(files)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i,j] = 1.0
            else:
                sim_matrix[i,j] = semantic_jaccard_similarity(
                    file_schemas[files[i]], file_schemas[files[j]], col_embeddings, sim_threshold
                )
    dist_matrix = 1 - sim_matrix

    df_dist = pd.DataFrame(
        sim_matrix,
        index=[os.path.basename(f) for f in files],
        columns=[os.path.basename(f) for f in files]
    )
    print(df_dist.iloc[:, :6].to_markdown())

    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram
    # Step 3.6: plot dendrogram
    linked = linkage(dist_matrix, method="average")

    plt.figure(figsize=(12, 6))
    dendrogram(
        linked,
        labels=[os.path.basename(f) for f in files],
        leaf_rotation=90,
        leaf_font_size=10,
        # color_threshold=0.3   # highlight early merges
    )
    plt.title("Agglomerative Clustering")
    plt.xlabel("Files")
    plt.ylabel("Distance")
    plt.ylim([-0.1, 3])
    plt.tight_layout()
    plt.savefig("imgs/cluster_dendrogram-semantic.svg")
    plt.savefig("imgs/cluster_dendrogram-semantic.pdf")
    # plt.show()

    # Step 4: clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average"
    )
    labels = clustering.fit_predict(dist_matrix)

    # Step 5: build cluster structure
    clusters = {}
    for label, file in zip(labels, files):
        clusters.setdefault(label, []).append(file)

    # Step 6: enrich cluster metadata
    clusters_out = {}
    for label, file_list in clusters.items():
        cluster_name = f"Cluster{label+1}"

        # schemas
        cluster_files = [{"file": f, "schema": list(file_schemas[f])} for f in file_list]

        # intersection schema
        intersection_schema = list(set.intersection(*(file_schemas[f] for f in file_list)))

        # average similarity
        sims = []
        for i in range(len(file_list)):
            for j in range(i+1, len(file_list)):
                sims.append(sim_matrix[files.index(file_list[i]), files.index(file_list[j])])
        avg_sim = float(np.mean(sims)) if sims else 1.0

        clusters_out[cluster_name] = {
            "files": cluster_files,
            "intersection_schema": intersection_schema,
            "average_similarity": avg_sim
        }

        # Print summary
        print(f"\n{cluster_name}:")
        print(f"  Intersection schema: {intersection_schema}")
        print(f"  Average similarity: {avg_sim:.2f}")
        for cf in cluster_files:
            print(f"    File: {cf['file']} Schema: {cf['schema']}")

    # Step 7: save to JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(clusters_out, f, indent=4)

    print(f"\nClusters saved to {output_file}")
    return clusters_out

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    base_path = "/home/data/bronze"
    clusters = semantic_file_clustering_jaccard(
        base_path,
        sim_threshold=0.6,
        distance_threshold=0.6,
        output_file="/home/data/silver/temp-semantic/cluster_schemas.json"
    )
