import os
import pandas as pd
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering


# -------------------------------
# Helper functions
# -------------------------------
def extract_schema(file_path):
    """
    Extract the schema (set of column names) from a file.

    Supports:
      - CSV: reads the first 5 rows and returns column names
      - Excel (xls/xlsx): iterates over all sheets and collects column names
      - JSON: checks top-level keys (for dict) or keys of the first few elements (for list of dicts)

    Args:
        file_path (str): path to the file

    Returns:
        set: set of column names
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, nrows=5)  # read only first rows for efficiency
            return set(df.columns)

        elif ext in [".xlsx", ".xls"]:
            xl = pd.ExcelFile(file_path)
            keys = set()
            for sheet in xl.sheet_names:  # process each sheet
                df = xl.parse(sheet, nrows=5)
                keys.update(df.columns)
            return keys

        elif ext == ".json":
            with open(file_path, "r") as f:
                data = json.load(f)
            keys = set()
            if isinstance(data, list):  # list of records
                for item in data[:5]:
                    if isinstance(item, dict):
                        keys.update(item.keys())
            elif isinstance(data, dict):  # single dict
                keys.update(data.keys())
            return keys

        else:
            return set()  # unsupported format

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return set()


def jaccard_distance_matrix(schema_list):
    """
    Compute the Jaccard distance matrix between all schemas.

    Jaccard similarity = |A ∩ B| / |A ∪ B|
    Jaccard distance    = 1 - similarity

    Args:
        schema_list (list[set]): list of schemas

    Returns:
        np.ndarray: square matrix of pairwise distances
    """
    n = len(schema_list)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            intersection = len(schema_list[i].intersection(schema_list[j]))
            union = len(schema_list[i].union(schema_list[j]))
            sim = intersection / union if union > 0 else 1.0
            dist_matrix[i, j] = 1 - sim  # distance
    return dist_matrix


# -------------------------------
# Main clustering function
# -------------------------------

def cluster_datalake_schemas_verbose(base_path, distance_threshold=0.5, output_file="clusters.json"):
    """
    Cluster file schemas in a data lake using agglomerative clustering
    with Jaccard distance.

    Args:
        base_path (str): root directory of the data lake
        distance_threshold (float): threshold for agglomerative clustering
        output_file (str): JSON file where clusters will be saved

    Returns:
        dict: clusters with file paths, schemas, and stats
    """
    file_paths = []
    schema_list = []

    # Step 1: Traverse the data lake and extract schemas
    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            schema = extract_schema(file_path)
            if schema:
                file_paths.append(file_path)
                schema_list.append(schema)

    if not schema_list:
        print("No schemas found in the data lake.")
        return {}

    # Step 2: Compute Jaccard distance matrix
    dist_matrix = jaccard_distance_matrix(schema_list)

    # Step 3: Cluster schemas using hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,  # let the algorithm decide number of clusters
        distance_threshold=distance_threshold,  # merge until this threshold
        metric="precomputed",  # we provide custom distance matrix
        linkage="average",  # average linkage for clusters
    )
    labels = clustering.fit_predict(dist_matrix)

    # Step 4: Group files by cluster label
    clusters = {}
    for label, file_path, schema in zip(labels, file_paths, schema_list):
        clusters.setdefault(label, []).append({"file": file_path, "schema": list(schema)})

    # Step 5: Prepare output structure with extra metadata
    output = {}
    for label, files in clusters.items():
        # Compute intersection schema across all files in cluster
        intersection = set.intersection(*(set(f["schema"]) for f in files))

        # Compute average Jaccard similarity within cluster
        n = len(files)
        if n > 1:
            sims = []
            for i in range(n):
                for j in range(i + 1, n):
                    inter = len(set(files[i]["schema"]).intersection(set(files[j]["schema"])))
                    union = len(set(files[i]["schema"]).union(set(files[j]["schema"])))
                    sims.append(inter / union if union > 0 else 1.0)
            avg_sim = float(np.mean(sims))
        else:
            avg_sim = 1.0  # only one file → perfect similarity

        output["Cluster" + str(label + 1)] = {
            "files": files,
            "intersection_schema": list(intersection),
            "average_similarity": avg_sim,
        }

        # Console output for traceability
        print(f"\nCluster {label+1}:")
        print(f"  Intersection schema: {intersection}")
        print(f"  Average Jaccard similarity: {avg_sim:.2f}")
        for f in files:
            print(f"    File: {f['file']}  Schema: {f['schema']}")

    # Step 6: Save results to JSON
    if not os.path.exists(output_file):
        os.makedirs(os.sep.join(output_file.split(os.sep)[:-1]), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nCluster schema information saved to {output_file}")
    return clusters


# -------------------------------
# Example usage
# -------------------------------
base_path = "/home/data/bronze"

clusters = cluster_datalake_schemas_verbose(
    base_path,
    distance_threshold=0.7,
    output_file="/home/data/silver/temp/cluster_schemas.json"
)
