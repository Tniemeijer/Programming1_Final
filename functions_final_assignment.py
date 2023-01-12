"""
Functions python file to keep the Jupyter notebook clean.
"""

import yaml
import pandas as pd


def yaml_config():
    """
    Open yaml config file in the same directory as the main script.
    ---------------------
    Output: config yaml as a python library.

    """
    with open(file="config.yaml",mode='r',encoding="utf8") as conf_file:
        config = yaml.safe_load(conf_file)
    return config

def load_concat_df(list_of_paths):
    """
    Use a list of paths, load csv and concaternate to a single dataframe.
    ---------------------
    Input: List with paths
    Output: Concaternated Dataframe
    """
    dfs = [ pd.read_csv(path,skiprows=13) for path in list_of_paths]
    df = pd.concat(dfs)
    return df

def make_stn_dict(stn_description_file):
    """
    Using a stn description txt file this function
    gets the coordinates per weatherstation (STN)
    ---------------------
    Input: path to weatherstation description file
    Output: dictionary with STN:"xx.xxN, xx.xxE , -/+ xxm"
    """
    with open(file=stn_description_file,mode='r',encoding="utf8") as stn_file:
        stn_coordinate_dict = {}
        for line in stn_file:
            if 'coordinates' in line:
                coordinates = line[16:43]
                coordinates = coordinates.replace(" ","")
                stn_number = stn_file.readline()[14:17]
                stn_coordinate_dict.update({stn_number:coordinates})
    return stn_coordinate_dict
