# src/crispdm/interpretation/cluster_interpreter_interpretation.py
"""Interpretation of clustering results – Phase 5.1.

Provides :func:`interpret_cluster_profiles` to translate raw profiling
data (feature importances per cluster) into human‑readable profiles
with top features and a short description.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from crispdm.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file from disk.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the JSON file.

    Returns
    -------
    Dict[str, Any]
        Deserialised content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_feature_names(schema_path: Path) -> List[str]:
    """Load a list of feature names from a schema report JSON.

    Supports common schema structures:
    * ``{"feature_names": [...]}``
    * ``{"features": [...]}``
    * ``{"column_names": [...]}``
    * A plain JSON list

    Parameters
    ----------
    schema_path : Path
        Path to the final schema report (created in Phase 3.5).

    Returns
    -------
    List[str]
        Ordered list of feature names.

    Raises
    ------
    ValueError
        If the schema is empty or does not contain a recognised key.
    """
    schema = load_json(schema_path)

    # Try common container keys
    for key in ("feature_names", "features", "column_names", "columns"):
        if key in schema and isinstance(schema[key], list):
            names = schema[key]
            if names:
                log.debug("Loaded %d feature names from key '%s'", len(names), key)
                return names
            else:
                raise ValueError(f"Feature names list under '{key}' is empty")

    # Fallback: the schema itself is a list
    if isinstance(schema, list):
        if schema:
            log.debug("Loaded %d feature names from top-level list", len(schema))
            return schema
        else:
            raise ValueError("Feature names list is empty")

    raise ValueError(
        f"Cannot extract feature names from schema at {schema_path}. "
        "Expected a dict with 'feature_names'/'features'/'column_names' or a list."
    )


def interpret_cluster_profiles(
    profiling_data: Dict[str, Any],
    feature_names: List[str],
    top_n: int = 10,
) -> Dict[str, Any]:
    """Build interpretable cluster profiles from raw profiling data.

    Parameters
    ----------
    profiling_data : Dict[str, Any]
        Expected to contain keys like ``"cluster_0"``, ``"cluster_1"`` etc.
        Each entry must have at least a ``"features"`` dict mapping
        feature index (string) → importance value.
        Optionally may include ``"size"`` and ``"centroid"``.
    feature_names : List[str]
        Ordered list of feature names matching the indices in ``features``.
    top_n : int, optional
        Number of top features to extract per cluster (default 10).

    Returns
    -------
    Dict[str, Any]
        Per‑cluster profiles with the structure::

            {
                "cluster_0": {
                    "top_features": [{"feature": "feat_name", "value": 0.5}, ...],
                    "size": 1234,
                    "description": "Cluster 0: Dominated by feat1, feat2, feat3"
                },
                ...
            }

    Raises
    ------
    ValueError
        If the input data is empty or malformed.
    """
    if not profiling_data:
        raise ValueError("profiling_data is empty – nothing to interpret.")

    profiles: Dict[str, Any] = {}

    for cluster_key, cluster_data in profiling_data.items():
        # Only process entries that look like cluster keys
        # if not cluster_key.startswith("cluster_"):
        #     log.warning("Skipping non-cluster key '%s'", cluster_key)
        #     continue

        try:
            int(cluster_key)  # "0", "1", "2" son válidos
        except ValueError:
            if not cluster_key.startswith("cluster_"):
                log.warning("Skipping non-cluster key '%s'", cluster_key)
                continue

        # features: Dict[str, float] = cluster_data.get("features", {})
        # if not features:
        #     log.warning("Cluster '%s' has no 'features' dict – skipping", cluster_key)
        #     continue
        # Sort features by absolute importance descending
        # sorted_items = sorted(
        #     features.items(),
        #     key=lambda x: abs(x[1]),
        #     reverse=True,
        # )

        # centroid_summary es lista de {"feature_index": int, "value": float}
        centroid_summary = cluster_data.get("centroid_summary", [])
        if not centroid_summary:
            log.warning("Cluster '%s' has no 'centroid_summary' – skipping", cluster_key)
            continue

        sorted_items = sorted(
            centroid_summary,
            key=lambda x: abs(x["value"]),
            reverse=True,
        )


        top_features = []
        # for idx_str, value in sorted_items[:top_n]:
        #     try:
        #         idx = int(idx_str)
        #     except (ValueError, TypeError):
        #         log.warning("Feature index '%s' is not an integer – skipping", idx_str)
        #         continue
        #
        #     feat_name = (
        #         feature_names[idx]
        #         if 0 <= idx < len(feature_names)
        #         else f"unknown_feature_{idx}"
        #     )
        #     top_features.append({
        #         "feature": feat_name,
        #         "value": round(value, 6),
        #     })
        for item in sorted_items[:top_n]:
            idx = item.get("feature_index")
            value = item.get("value", 0.0)
            if idx is None:
                continue
            feat_name = feature_names[idx] if 0 <= idx < len(feature_names) else f"unknown_feature_{idx}"
            top_features.append({"feature": feat_name, "value": round(value, 6)})

        cluster_id = cluster_key.split("_", 1)[1] if "_" in cluster_key else cluster_key
        description = _generate_description(cluster_id, top_features)

        profiles[cluster_key] = {
            "top_features": top_features,
            "size": cluster_data.get("size"),
            "description": description,
        }

    if not profiles:
        raise ValueError("No valid cluster profiles could be extracted.")

    return profiles


def _generate_description(
    cluster_id: str,
    top_features: List[Dict[str, Any]],
) -> str:
    """Generate a short, human‑readable description for a cluster.

    Parameters
    ----------
    cluster_id : str
        Cluster identifier (e.g., ``"0"``, ``"1"``).
    top_features : List[Dict[str, Any]]
        Top features with ``"feature"`` and ``"value"`` keys.

    Returns
    -------
    str
        Natural‑language description.
    """
    if not top_features:
        return f"Cluster {cluster_id}: No dominant features"

    top_three = [f["feature"] for f in top_features[:3]]
    return f"Cluster {cluster_id}: Dominated by {', '.join(top_three)}"