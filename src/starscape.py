'''
Randomly generates a starscape.
'''

import numpy as np
import os
import random
import sys

import process as proc
import utility as util
import formula as f

# parameters
img_size = (32, 1024, 1024)
feature_size = (64.0, 128.0, 128.0)
chunk_size = (32, 32, 32)

def seed():
    # get seed value from user
    s = input("Enter an integer seed (default: random): ")
    # default case
    if s == '':
        s = np.random.randint(2**32 - 1)
        print("Using seed", s)
    else:
        s = int(s)
    # generate a random seed for the rest of computation
    random.seed(s)
    np.random.seed(s)
    return s

def prob(s):
    # calculate a new probability map if the user requests it
    p = np.array([])
    path = input("Enter file path to raw probability file (default: {:s}): ".format(os.path.join(os.path.curdir, 'output', 'prob.raw')))
    # default case
    if path == '':
        path = os.path.join(os.path.curdir, 'output', 'prob.raw')
    # attempt to load file
    try:
        p = np.fromfile(path)
        p = np.reshape(p, img_size)
        print("Using existing file")
    # if file does not exist, generate a new starscape
    except OSError:
        print("File not found. Generating a new starscape.")
        p = proc.probability_map(s, img_size)
        p.tofile(path)

    # get stringing value from user
    str = input("Enter a probability reduction value (default: 2): ")
    # default case
    if str == '':
        str = 2
    else:
        str = int(str)

    # modify probability map by the requested power
    p = np.power(p, str)
    util.write_dist_img(p, img_size, os.path.join(os.path.curdir, 'output', 'distribution'))
    return p

def cluster(prob):
    # get clustering amout
    clusval = input("Enter cluster cutoff, in the range 0-1 (default: 0.7): ")
    if clusval == '':
        clusval = 0.7
    else:
        clusval = float(clusval)

    a = input("Enter the age of the universe, in billions of years (default: 13.8): ")
    if a == '':
        a = 13.8
    else:
        a = int(a)

    # determine cluster locations
    clusters, ages = proc.find_clusters(prob, clusval, a, img_size)
    util.write_cluster_image(clusters, img_size, os.path.join(os.path.curdir, 'output', 'clusters'))
    return clusters, ages, a

def star(prob, clusters):
    count = input("Enter the number of stars to generate (default: 15000): ")
    if count == '':
        count = 15000
    else:
        count = int(count)

    # generate stars in space
    return proc.generate_stars(prob, clusters, count, img_size)


if __name__ == '__main__':
    s = seed()
    p = prob(s)
    c, a, max_age = cluster(p)
    stars = star(p, c)
    stars = proc.age_stars(stars, a, max_age)

    # draw images and generate a HR diagram
    util.write_HR_diagram(stars, os.path.join(os.path.curdir, 'output', 'HR'))
    util.write_star_image(stars, img_size, os.path.join(os.path.curdir, 'output', 'stars_eye'), distance=5)
    util.write_star_image(stars, img_size, os.path.join(os.path.curdir, 'output', 'stars_hubble'), distance=1000)
