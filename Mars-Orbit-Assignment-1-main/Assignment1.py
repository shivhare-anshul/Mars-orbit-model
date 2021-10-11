import numpy as np
import pandas as pd
from math import cos, sin, tan, sqrt, atan, radians, degrees
# import math
# from datetime import timedelta

## Pre-processing  of input data

# function to compute time-differences by taking oppositions pandas dataframe as input
def get_time_diff(data):
    timediff = [0]
    for i in range(1, 12):
        diff = data.iloc[i]['time'] - data.iloc[i-1]['time']
        diff_days = diff.total_seconds() / (60 * 60 * 24)
        timediff.append(diff_days)
    return np.array(timediff)

# function to compute longitudes by taking oppositions pandas dataframe as input
def get_longitudes(data):
    long = []
    for i in range(12):
        row = data.iloc[i]
        longitude = row['ZodiacIndex'] * 30 + row['Degree'] + row['Minute.1'] / 60 + row['Second'] / 3600
        long.append(longitude)
    return np.array(long)

# function to take the name and path of the csv file as input and returns the time-differences and longitudes as vertically stacked array
def get_data(filename):
    opp = pd.read_csv(filename)
    opp['year'] = opp['Year'] + 400
    opp['time'] = pd.to_datetime(opp[['year', 'Month', 'Day', 'Hour', 'Minute']])
    timediff = get_time_diff(opp)
    long = get_longitudes(opp)
    # vertically stack "timediff" and "long" in order to pass it later as a single parameter
    data = np.vstack((timediff, long))
    return data

# computes the angular error delta between the point of intersection with the circle and the heliocentric longitude of the opposition 
# More explanation of this function is written in the attached report file
def find_delta(x, y, longitude):
    angle = 0
    # adjusting the inverse tan function for different quadrants
    if x > 0:
        if y > 0:
            angle = atan(y / x)
        else:
            y = -y
            angle = radians(360) - atan(y / x)
    else:
        x=-x
        if y > 0:
            angle = radians(180) - atan(y / x)
        else:
            y=-y
            angle = radians(180) + atan(y / x)
    return degrees(abs(angle - longitude))

## First question --> compute error for given inputs
def MarsEquantModel(c, r, e1, e2, z, s, data):
    # store angles in radians for simplicity
    c = radians(c)
    e2 = radians(e2)
    z = radians(z)
    errors = np.zeros(12)

    timediff = data[0]
    long = data[1]

    # iterate through the oppositions and compute the error
    for i in range(12):
        z = degrees(z)
        z += (s * timediff[i])
        z = z % 360
        z = radians(z)

        # y = m1 * x + c1 is the equation of the first line intersecting the circular orbit, where m1 = tan(z)
        c1 = e1 * sin(e2) - e1 * cos(e2) * tan(z)

        # need to solve a quadratic equation Ax^2 + Bx + C = 0 to get point of intersection (x,y) 
        A = 1 + tan(z) * tan(z)
        B = 2 * c1 * tan(z) - 2 * cos(c) - 2 * tan(z) * sin(c)
        C = c1 * c1 - 2 * c1 * sin(c) + 1 - r * r

        # solve the quadratic equation to get x-coordinates of point of intersection
        disc = B * B - 4 * A * C
        x1 = (-B + sqrt(disc)) / (2 * A)
        x2 = (-B - sqrt(disc)) / (2 * A)
        # now get the y-coordinates
        y1 = x1 * tan(z) + c1
        y2 = x2 * tan(z) + c1 
        # (there are two roots as the line may intersect the circle at upto two points)

        # after finding the point of intersection, use find_delta function to compute the angular error delta and choose the minimum of both
        delta1 = find_delta(x1, y1, radians(long[i]))
        delta2 = find_delta(x2, y2, radians(long[i]))

        min_delta = min(delta1, delta2)
        errors[i] = min_delta

    max_error = max(errors)
    return errors, max_error


## Second question --> Fix r & s and search others
def bestOrbitInnerParams(r,s, data):
    best_params = dict()
    best_error = 999999
    best_opp_errors = []
    long = data[1]

    for c in np.arange(130, 160, 0.1): # add tqdm here if necessary
        for e2 in np.arange(c - 20, c + 20, 0.1):
            for e1 in np.arange(0.5, 2.1, 0.1):
                for z in np.arange(long[0] - 15, long[0] + 15, 0.1):
                    opp_errors, error = MarsEquantModel(c=c, r = r, e1 = e1, e2 = e2, z = z, s = s, data = data)
                    if error < best_error:
                        best_error = error
                        best_opp_errors = opp_errors
                        best_params = {'c':c, 'e2':e2, 'e1':e1, 'z':z}

    return best_params['c'], best_params['e1'], best_params['e2'], best_params['z'], best_opp_errors, best_error


## Third question --> Fix r and search for s
def bestS(r, data):
    best_params = dict()
    best_error = 999999
    best_opp_errors = []

    search_space = np.arange(350/687, 370/687, 0.1/687)

    for s in search_space:
        c, e1, e2, z, opp_errors, err = bestOrbitInnerParams(r, s, data)
        if err < best_error:
            best_error = err
            best_opp_errors = opp_errors
            best_params = {'c':c, 'e1':e1, 'e2':e2, 'z':z, 's':s}
    
    return best_params['s'], best_opp_errors, best_error


## Fourth question --> Fix s and search for r
def bestR(s, data):
    best_params = dict()
    best_error = 999999
    best_opp_errors = []

    # For more precise values of R, narrow down the search space and use increment of 0.01.

    for r in np.arange(6, 12, 0.1):
        c, e1, e2, z, opp_errors, err = bestOrbitInnerParams(r, s, data)
        if err < best_error:
            best_error = err
            best_opp_errors = opp_errors
            best_params = {'c':c, 'e1':e1, 'e2':e2, 'z':z, 'r':r}
    
    return best_params['r'], best_opp_errors, best_error


## Fifth question --> Search both r and s

def bestMarsOrbitParams(data):
    best_params = dict()
    best_error = 999999
    best_opp_errors = []
    
    s_search_space = np.arange(355/687, 365/687, 0.1/687)  # 0.1 / 687 increment, so will take a long time to execute
    r_search_space = np.arange(7, 11, 0.1)  # 0.1 increment

    for s in s_search_space:
        for r in r_search_space:
            c, e1, e2, z, opp_errors, err = bestOrbitInnerParams(r, s, data)
            if err < best_error:
                best_error = err
                best_opp_errors = opp_errors
                best_params = {'c':c, 'e1':e1, 'e2':e2, 'z':z, 'r':r, 's':s}
            
    return best_params['r'], best_params['s'], best_params['c'], best_params['e1'], best_params['e2'], best_params['z'], best_opp_errors, best_error


# Test error for optimal parameters
# (c, r, e1, e2, z, s, data)

# Uncomment and run below line to convert the csv data input into numpy array data of size 2 x 12
# data = get_data('01_data_mars_opposition_updated.csv')

# Uncomment and run below line to test the function
# errors, maxError = MarsEquantModel(139.0, 8.732, 1.60, 148.70, 56.47, 360 / 687, data)

# errors, maxError = MarsEquantModel(150, 9.0, 1.4, 150, 54, 360 / 687, data)

# c, e1, e2, z, errors, maxError = bestOrbitInnerParams(8.732, 360 / 687, data)

# s, errors, maxError = bestS(8.732, data)

# r, errors, maxError = bestR(360 / 687, data)

# r, s, c, e1, e2, z, errors, maxError = bestMarsOrbitParams(data)

# (c, r, e1, e2, z, s, data)
# print(maxError)
# print(errors)

