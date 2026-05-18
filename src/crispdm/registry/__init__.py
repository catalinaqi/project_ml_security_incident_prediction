# Artifact generator registry package.
# Exposes the dispatch engine and activates all phase generators on import.
from crispdm.registry.generator_registry_registry import (
    register_artifact,
    write_output_artifacts,
)
from crispdm.registry import (
    phase2_generator_registry,
    phase3_generator_registry,
    phase4_generator_registry,
    phase5_generator_registry,
)