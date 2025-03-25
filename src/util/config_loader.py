import json
import os
from pathlib import Path
from typing import Any


class ConfigLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_config(config_path: str) -> dict[str, Any]:
        assert Path(config_path).is_file(), f"File '{config_path}' is a directory / does not exist!"
        assert os.path.splitext(config_path)[1] == ".json", f"File '{config_path}' is not a JSON file!"
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
