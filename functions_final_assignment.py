"""
Functions python file to keep the Jupyter notebook clean.
"""

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, gamma, iqr
import re


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
                coordinates = line[16:32]
                coordinates = coordinates.replace(" ","")
                coordinates = re.sub(string=coordinates,pattern="N|E",repl="")
                coordinates = [float(x) for x in coordinates.split(',')]
                stn_number = stn_file.readline()[14:17]
                stn_coordinate_dict.update({stn_number:coordinates})
    return stn_coordinate_dict


def hist_robust_dist(dataframe, distribution = 'norm'):
    """
    Histogram with normal distribution using the robust method.
    Borrowed pieces of code from the DS1 statistics functions from Emile Apol.
    User can choose the type of distribution:
        'norm' = normal distribution (scipy.stats.norm)
        'gamma' = gamma distribution (scipy.stats.norm)

    >>> hist_robust_norm({'A':10,'B':20,'C':30}, distribution = 'gamma' or 'norm')
        if gamma is chosen user must supply. 

    ---------------------
    Input: Dataframe

    Ouput: histgram with plotted normal distribution and robust mu and sigma.

    """
    mu = np.mean(dataframe)
    sigma = iqr(dataframe)/1.349

    x = np.linspace(np.min(dataframe), np.max(dataframe), 501)
    match distribution:
        case 'norm':
            rv = np.array([
                norm.pdf(xi, loc = mu, scale = sigma) for xi in x
                ])
        case 'gamma':
            rv = np.array([
                gamma.pdf(xi, a=6.5, loc = 1000, scale = 300) for xi in x
                ])
    print(f'mu (robust) = {mu:.5}, sigma (robust) = {sigma:.5}')
    plt.hist(dataframe, density=True,
     bins='auto', alpha=1, rwidth=1, label="experimental", color="grey")
    plt.plot(x, rv, 'r', label='distibrution approximation')
    plt.axvline(mu, color="red", label="Mu")
    plt.ylabel("Probability")
    plt.xlabel("Precipitation(x0.1mm)")
    plt.title("Histogram with distribution robust mean")
    plt.legend()
    plt.show()


def validate_precipitation(x,lower,upper):
    """
    Function checks if x (input precipitation value) is higher or lower than
    the upper and lower bounds.

    >>> validate_precipitation(x=200,lower=100,upper=300)

    ---------------------
    Input: x (numerical value, either int or float)

    Output: string with either "wet","dry" or "normal"

    """
    if x < lower:
        return "dry"
    elif x > upper:
        return "wet"
    else:
        return "normal"

def read_geojson(path):
    """
    Opens geojson and returns dataframe with the properties.

    >>> read_geojson("~/location/to/file.json")

    -------------------
    Input: path (string)
    Output: pd.Dataframe
    """
    geo_df = pd.read_json(path)
    geo_df = pd.json_normalize(geo_df["features"])
    return geo_df

def get_coordinates_df(nested_list_coord):
    """
    From a nested geojson coordinates list, generates a dataframe with
    x and y coordinates. So use this per row in the coordinates column.

    >>> get_coordinates([[[52.12,14.56],[52.45,14.66],[52.32,14.60]]])

    -------------------
    Input: nested geometry.coordinates data

    Output: Dataframe with x and y columns.
    """
    xd = []
    yd = []
    for l in nested_list_coord:
        for i in l:
            xd.append(i[0])
            yd.append(i[1])
    return pd.DataFrame({'x':xd,'y':yd})

def get_centerpoint(dataframe_coordinates):
    """
    Source: https://stackoverflow.com/questions/4355894/how-to-get-center-of-set-of-points-using-python
    Function calculates the centroid of the inserted.

    """
    data = get_coordinates_df(dataframe_coordinates)

    x = data.x.to_numpy() 
    y = data.y.to_numpy()

    #print(f'{x,y}')

    N = range(len(data)-1)
    M = np.array([(x[i]-x[i+1])*(y[i]+y[i+1])/2 for i in N]) #Area of each trapezoid
    My = np.array([(x[i]+x[i+1])/2 for i in N])*M #Moment of area (area*distance to centroid) with respect to the Y axis of each trapezoid
    Mx = np.array([(y[i]+y[i+1])/4 for i in N])*M #Moment of area (area*distance to centroid) with respect to the X axis of each trapezoid
    X = sum(My)/sum(M)
    Y = sum(Mx)/sum(M)
    centroid = [X,Y]
    points_ave = data.mean(axis=0)

    plt.plot(data.x, data.y, 'r',marker='.',markeredgecolor='black', markersize=3)
    plt.plot(*centroid, 'blue', marker='o',markeredgecolor='black', markersize=7)
    plt.plot(*points_ave, 'green', marker='o',markeredgecolor='black', markersize=7)
    plt.axis('equal')
    plt.xlim((4, 6))
    plt.ylim((51,53))
    plt.legend(['line','Centroid','Average of points'])
    return [Y,X]

def calc_point_dist(vector1,vector2):
    """
    Function calculates the difference between points using Pythagorean theorem.
    """
    np1,np2 = np.array(vector1), np.array(vector2)
    dif  = np1 - np2
    return np.sqrt((dif[0]**2)+(dif[1]**2))
