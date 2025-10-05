import os
import json
import pandas as pd
# from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# Load semantic embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------------
# Load data utility
# -------------------------------
def load_dataframe(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            return pd.read_csv(file_path)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        elif ext == ".json":
            data = pd.read_json(file_path)
            if isinstance(data, dict):
                return pd.DataFrame([data])
            return data
        else:
            print(f"Unsupported format: {file_path}")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def compute_embeddings(columns):
    """Compute embeddings for a list of column names."""
    return {col: model.encode(col, convert_to_tensor=True, normalize_embeddings=True) for col in columns}

def hybrid_similarity(col1, col2, emb1, emb2, alpha=0.5):
    """Compute hybrid similarity using fuzzy + semantic embeddings."""
    fuzzy_score = 1 # fuzz.token_sort_ratio(col1, col2) / 100.0
    semantic_score = util.cos_sim(emb1, emb2).item()
    return alpha * fuzzy_score + (1 - alpha) * semantic_score


def suggest_schema_mapping(schemas, threshold=0.7, alpha=0.5):
    """
    Create mapping of original -> unified names using hybrid similarity.
    Returns:
      - mapping: {original_column: unified_column}
      - groups: {unified_column: [all original variants]}
    """
    all_columns = set().union(*schemas.values())
    embeddings = compute_embeddings(all_columns)
    
    canonical = {}
    groups = {}

    for col in all_columns:
        if not canonical:
            canonical[col] = col
            groups[col] = [col]
        else:
            # compute similarity with existing canonical keys
            scores = {c: hybrid_similarity(col, c, embeddings[col], embeddings[c], alpha=alpha)
                      for c in canonical.keys()}
            best_match, best_score = max(scores.items(), key=lambda x: x[1])
            if best_score >= threshold:
                unified = canonical[best_match]
                canonical[col] = unified
                groups[unified].append(col)
            else:
                canonical[col] = col
                groups[col] = [col]

    return canonical, list(groups.keys()), groups



# -------------------------------
# Unify + merge
# -------------------------------
def unify_and_merge(cluster_files, cluster_id, output_dir, threshold=0.7, alpha=0.5):
    schemas = {}
    dataframes = {}
    for file in cluster_files:
        df = load_dataframe(file)
        if df is not None:
            schemas[file] = set(df.columns)
            dataframes[file] = df

    if not schemas:
        print(f"No valid data in {cluster_id}")
        return

    mapping, unified_columns, groups = suggest_schema_mapping(schemas, threshold=threshold, alpha=alpha)

    transformed_dfs = []
    for file, df in dataframes.items():
        # Step 1: Rename using canonical mapping
        rename_map = {col: mapping[col] for col in df.columns if col in mapping}
        df = df.rename(columns=rename_map)

        # Step 2: Collapse duplicates inside this file
        for unified_col, variants in groups.items():
            existing = [c for c in variants if c in df.columns]
            if len(existing) > 1:
                # collapse into one column (take first non-null)
                df[unified_col] = df[existing].bfill(axis=1).iloc[:, 0]
                df.drop(columns=[c for c in existing if c != unified_col], inplace=True, errors="ignore")
            elif len(existing) == 1:
                # rename single variant to unified_col
                if existing[0] != unified_col:
                    df = df.rename(columns={existing[0]: unified_col})

        # After all collapsing, ensure columns are unique
        df = df.loc[:, ~df.columns.duplicated()]
        transformed_dfs.append(df)

    # Step 3: Concatenate all files in cluster
    unified_df = pd.concat(transformed_dfs, ignore_index=True, sort=False)

    # Step 4: Reorder columns consistently
    unified_df = unified_df.reindex(columns=unified_columns)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{cluster_id}.csv")
    unified_df.to_csv(output_file, index=False)
    print(f"{cluster_id} merged into {output_file}")

    schema_file = os.path.join(output_dir, f"{cluster_id}_schema.json")
    with open(schema_file, "w") as f:
        json.dump(
            {"mapping": mapping, "groups": groups, "unified_columns": unified_columns},
            f,
            indent=4,
        )
    print(f"Schema mapping saved to {schema_file}")



# -------------------------------
# Main function
# -------------------------------
def merge_clusters(cluster_file, output_dir="merged_clusters", threshold=0.7, alpha=0.5):
    with open(cluster_file, "r") as f:
        clusters = json.load(f)

    for cluster_name, cluster_data in clusters.items():
        cluster_files = [f["file"] for f in cluster_data["files"]]
        print(f"\nProcessing {cluster_name} with {len(cluster_files)} files...")
        unify_and_merge(cluster_files, cluster_name, output_dir, threshold=threshold, alpha=alpha)


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    cluster_file = "/home/1-profiling/data/silver/temp-semantic/cluster_schemas.json"
    merge_clusters(
        cluster_file,
        output_dir="/home/1-profiling/data/silver/temp-semantic/unified_clusters",
        threshold=0.6,
        alpha=0  # more weight to semantic similarity
    )
