"""
Functions python file to keep the Jupyter notebook clean.
"""

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, iqr


def yaml_config():
    """
    Opens config.yaml in the same directory as the main script.
    ---------------------
    Output: config yaml as a python library.
    """
    with open(file="config.yaml",mode='r',encoding="utf8") as conf_file:
        config = yaml.safe_load(conf_file)
    return config

def load_concat_df(list_of_paths):
    """
    Use a list of paths, load csv and concaternate to a single dataframe.
    >>> load_concat_df(["//path", "//path"])

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


def hist_robust_norm(dataframe):
    """
    Histogram with normal distribution using the robust method.
    Borrowed pieces of code from the DS1 statistics functions from Emile Apol.
    """
    mu = np.mean(dataframe)
    sigma = iqr(dataframe)/1.349

    x = np.linspace(np.min(dataframe), np.max(dataframe), 501)
    rv = np.array([norm.pdf(xi, loc = mu, scale = sigma) for xi in x])

    print(f'mu (robust) = {mu:.5}, sigma (robust) = {sigma:.5}')
    plt.hist(dataframe, density=True,
     bins='auto', alpha=1, rwidth=1, label="experimental")
    plt.plot(x, rv, 'r', label='Normal approximation')
    plt.axvline(mu, color="red", label="Mu")
    plt.ylabel("Probability")
    plt.xlabel("Precipitation(x0.1mm)")
    plt.title("Histogram normal distribution (robust)")
    plt.legend()
    plt.show()
