# src/crispdm/model/algorithm_selector_model.py
"""CRISP-DM Phase 4.1 – Algorithm Selection logic.

This module contains pure functions that interpret configuration
and return selection metadata.  No file I/O, no side effects.
"""

from __future__ import annotations

from typing import Any

from crispdm.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def select_algorithms(
        methods_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return a list of selected algorithm descriptors from configuration.

    Parameters
    ----------
    methods_cfg : dict[str, Any]
        The ``methods.algorithm_selection`` section of the step config.

    Returns
    -------
    list[dict[str, Any]]
        Each dict contains:
            - ``name`` (str): algorithm key (e.g. ``"kmeans"``)
            - ``enabled`` (bool)
            - ``priority`` (int)
            - ``output`` (str): relative artifact path
            - ``params`` (dict): any additional parameters (e.g. grid ranges)
    """
    techniques: dict[str, Any] = methods_cfg.get("techniques", {})
    if not techniques:
        log.warning("[algorithm_selector] 'techniques' dict is empty or missing")
        return []

    selected: list[dict[str, Any]] = []
    for algo_name, algo_cfg in techniques.items():
        enabled = algo_cfg.get("enabled", False)
        if not enabled:
            log.debug("[algorithm_selector] algorithm '%s' disabled – skipping", algo_name)
            continue

        entry = {
            "name": algo_name,
            "enabled": True,
            "priority": algo_cfg.get("priority", 999),
            "output": algo_cfg.get("output", ""),
            "params": algo_cfg.get("params", {}),
        }
        selected.append(entry)
        log.info("[algorithm_selector] selected algorithm '%s' with priority %d",
                 algo_name, entry["priority"])

    if not selected:
        log.warning("[algorithm_selector] no algorithm selected – all disabled or empty")

    return selected