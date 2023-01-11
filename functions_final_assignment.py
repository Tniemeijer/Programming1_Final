"""
Functions python file to keep the Jupyter notebook clean.
"""

import yaml

def yaml_config():
    """
    Open yaml config file in the same directory as the main script.
    ---------------------
    Output: config yaml as a python library.

    """
    with open(file="config.yaml",mode='r',encoding="utf8") as conf_file:
        config = yaml.safe_load(conf_file)
    return config
