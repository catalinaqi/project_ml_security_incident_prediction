# src/crispdm/model/test_design_reporter_model.py
"""Pure model function for CRISP-DM Phase 4.4 – Test Design Generation.

Produces an evaluation plan dictionary from the step configuration.
No side effects (no I/O, no logging).  # <-- se añade logging (NO recomendado)
"""

from __future__ import annotations

from typing import Any

from crispdm.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def generate_test_design(step_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return an evaluation plan dictionary based on the step configuration.

    Parameters
    ----------
    step_cfg : dict
        Configuration for ``step_4_4_test_design`` from the pipeline YAML.
        Expected keys (some optional):
        - ``description`` (str)
        - ``metrics`` (list of str)
        - ``validation_approach`` (str)
        - ``output`` (str) – fallback output path
        - ``output_artifacts`` (dict) – preferred output path source

    Returns
    -------
    dict
        Evaluation plan with keys:
        - ``description``
        - ``metrics``
        - ``validation_approach``
        - ``output_path``
        - ``expected_artifacts``
        - ``note``
    """
    log.debug("[generate_test_design] creating evaluation plan")

    # Extract configuration values with sensible defaults
    description = step_cfg.get("description", "")
    metrics = step_cfg.get("metrics", [])
    validation_approach = step_cfg.get("validation_approach", "")
    output_path = (
        step_cfg.get("output_artifacts", {}).get("evaluation_plan")
        or step_cfg.get("output")
        or "4.4.test_design.evaluation_plan.json"
    )

    log.debug("[generate_test_design] metrics=%s, approach=%s", metrics, validation_approach)

    if not metrics:
        log.warning("[generate_test_design] no metrics defined in configuration – plan will be empty")

    if not validation_approach:
        log.warning("[generate_test_design] no validation_approach defined in configuration")

    # Expected artifacts that downstream steps (4.5) will produce
    expected_artifacts = [
        "silhouette.json",
        "davies_bouldin.json",
        "calinski_harabasz.json",
        "adjusted_rand_index.json",
        "cluster_profiling.json",
    ]

    plan = {
        "description": description,
        "metrics": metrics,
        "validation_approach": validation_approach,
        "output_path": output_path,
        "expected_artifacts": expected_artifacts,
        "note": (
            "This plan documents the evaluation strategy. "
            "Actual metrics and validation are computed in step 4.5."
        ),
    }

    log.info("[generate_test_design] evaluation plan generated – path=%s", output_path)
    return plan