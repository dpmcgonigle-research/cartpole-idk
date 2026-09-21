"""Small analytics result writers (configuration is serialized by Pydantic)."""

from cartpole_idk.artifacts.io import software_versions, write_json, write_table

__all__ = ["software_versions", "write_json", "write_table"]
