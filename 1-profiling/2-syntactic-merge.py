import os
import json
import pandas as pd
from rapidfuzz import fuzz, process


def suggest_schema_mapping(schemas, threshold=70):
    """
    Suggest a unified schema mapping given multiple schemas.
    Uses fuzzy string matching to align column names.
    """
    all_columns = set().union(*schemas.values())
    canonical = {}
    for col in all_columns:
        if not canonical:
            canonical[col] = col
        else:
            match, score, _ = process.extractOne(col, canonical.keys(), scorer=fuzz.token_sort_ratio)
            if score >= threshold:
                canonical[col] = canonical[match]  # align to existing canonical name
            else:
                canonical[col] = col
    return canonical


def unify_datasets(file_schemas, mapping, dataframes):
    """
    Apply schema mapping to actual dataframes and unify them.
    """
    transformed_dfs = []
    for file, df in dataframes.items():
        new_cols = {col: mapping[col] for col in df.columns if col in mapping}
        df = df.rename(columns=new_cols)
        transformed_dfs.append(df)
    unified_df = pd.concat(transformed_dfs, ignore_index=True, sort=False)
    return unified_df


def build_unified_clusters(cluster_file, output_dir="unified_clusters"):
    """
    Read cluster JSON file, suggest schema for each cluster,
    and build unified DataFrames with values from the files.
    """
    with open(cluster_file, "r") as f:
        clusters = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    unified_results = {}
    for cluster_id, info in clusters.items():
        print(f"\nProcessing Cluster {cluster_id}...")
        # Collect schemas
        file_schemas = {f["file"]: set(f["schema"]) for f in info["files"]}
        mapping = suggest_schema_mapping(file_schemas, threshold=70)
        print(f"  Suggested mapping: {mapping}")

        # Load DataFrames
        dataframes = {}
        for f in info["files"]:
            file_path = f["file"]
            ext = os.path.splitext(file_path)[1].lower()
            try:
                if ext == ".csv":
                    df = pd.read_csv(file_path)
                elif ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(file_path)
                elif ext == ".json":
                    df = pd.read_json(file_path)
                else:
                    continue
                dataframes[file_path] = df
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")

        if not dataframes:
            print("  No valid files loaded.")
            continue

        # Unify into one DataFrame
        unified_df = unify_datasets(file_schemas, mapping, dataframes)

        # Save unified dataset
        out_path = os.path.join(output_dir, f"{cluster_id}.csv")
        unified_df.to_csv(out_path, index=False)
        print(f"  Unified dataset saved to {out_path}")

        unified_results[cluster_id] = {
            "mapping": mapping,
            "unified_file": out_path,
            "num_records": len(unified_df),
        }

    return unified_results


# -------------------------------
# Example usage
# -------------------------------
cluster_file = "/home/1-profiling/data/silver/temp/cluster_schemas.json"  # generated earlier
results = build_unified_clusters(cluster_file, output_dir="/home/1-profiling/data/silver/temp/unified_clusters")
print("\nSummary of unified clusters:")
print(json.dumps(results, indent=4))
