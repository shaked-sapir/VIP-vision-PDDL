from pathlib import Path

import yaml


def load_config(config_path: Path = None) -> dict:
    """
    Load configuration from YAML file.

    :param config_path: Path to config file. If None, uses default 'config.yaml' in project root.
    :return: Configuration dictionary.
    """
    if config_path is None:
        # Find project root (where config.yaml should be)
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # Go up from src/ to project root
        config_path = project_root / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}.\n"
            f"Please copy config.example.yaml to config.yaml and fill in your values."
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base_dir = config_path.parent

    # Normalize all path-like entries (recursive)
    def resolve_paths(d):
        if isinstance(d, dict):
            return {k: resolve_paths(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [resolve_paths(v) for v in d]
        elif isinstance(d, str) and ("/" in d or "\\" in d):
            p = Path(d)
            if not p.is_absolute():
                return str((base_dir / p).resolve())
            return str(p)
        else:
            return d

    config = resolve_paths(config)

    return config
