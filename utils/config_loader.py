import yaml
import os

class Config2Struct:
    # Convert the config dictionary into an object or class: 
    # All keys from the YAML file are directly mapped as attributes of the Config2Struct object
    # # and iterate by the get_config_params(obj, level=0) function (below) .
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_all_configs(general_config_path):
    """Loads the general config and merges submodule configs."""
    # Load general configuration
    general_config = load_config(general_config_path)

    # Load submodule configurations
    submodule_configs = {}
    for submodule, submodule_path in general_config["submodule_configs"].items():
        submodule_configs[submodule] = load_config(submodule_path)

    # Combine general and submodule configs
    return {"general": general_config, **submodule_configs}

def get_config_params(obj, level=0):
    indent = "  " * level  # Indentation for nested levels
    if hasattr(obj, "__dict__"):  # If the object has attributes
        for key, value in vars(obj).items():
            if hasattr(value, "__dict__"):  # Nested Config2Struct
                print(f"{indent}{key}: (nested object)")
                get_config_params(value, level + 1)
            elif isinstance(value, dict):  # Dictionary inside the config
                print(f"{indent}{key}: (nested dictionary)")
                for subkey, subvalue in value.items():
                    print(f"{indent}  {subkey}: {subvalue}")
            else:  # Base case: simple attributes
                print(f"{indent}{key}: {value}")
    else:
        print(f"{indent}{obj}")
