'''
Randomly generates a starscape.
'''

from concurrent.futures import ThreadPoolExecutor
import numpy as np
from opensimplex import OpenSimplex
import os
import random
import sys
import time

import utility as util
import formula as f

# parameters
feature_size = (64, 128.0, 128.0)
chunk_size = (32, 32, 32)

# internal representation of a star
class Star:
    def __init__(self, pos, type, mass, cluster):
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        self.type = type
        self.mass = mass
        self.cluster = cluster
        self._lum = f.lum(self.mass)
        self._temp = f.temp(self.mass)
        self.age = 0
        self._color = ()
    # get this star's position
    def pos(self):
        return (self.x, self.y, self.z)
    # get this star's luminosity
    def lum(self):
        return self._lum
    # If we use the realistic luminosity for a star when generating an image, we will
    # only be able to see the O-type stars. We use this method to get "luminosity"
    # in the range 0-9, so images look better.
    def img_lum(self):
        return  "-MK--GFABO".index(self.type)
    # get this star's temperature
    def temp(self):
        return self._temp
    # get this star's color
    def color(self):
        if self._color == ():
            self._color = f.color(self.temp())
        return self._color

# map location to probability of star forming there
def prob_worker(vals):
    t = time.time()
    # get values passed in
    x, y, z, seed = vals
    # seed simplex generator
    simplex = OpenSimplex(seed)
    # initialize data chunk
    chunk = np.zeros(chunk_size).astype(float)
    # iterate over chunk
    for i in range(chunk_size[0]):
        for j in range(chunk_size[1]):
            for k in range(chunk_size[2]):
                # generate data
                chunk[i,j,k] = simplex.noise3d(((x*chunk_size[0]) + i) / feature_size[0],
                                               ((y*chunk_size[1]) + j) / feature_size[1],
                                               ((z*chunk_size[2]) + k) / feature_size[2])
    return x, y, z, chunk
def probability_map(seed, img_size):
    # initialize data structures
    prob = np.zeros(img_size)
    t = time.time()
    workers = []
    threads = os.cpu_count()
    print("Generating probability map with {:d} threads... 0.00%".format(threads), flush=True, end=" ")

    chunks = (np.asarray(img_size) // np.asarray(chunk_size)).astype(int)
    # generate chunks in parallel
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # add all chunks to the thread pool
        for cx in range(chunks[0]):
            for cy in range(chunks[1]):
                for cz in range(chunks[2]):
                    workers.append(executor.submit(prob_worker, (cx, cy, cz, seed)))
        # wait for the chunks to complete
        percent = 0.0
        while len(workers) != 0:
            for w in workers:
                if w.done():
                    cx, cy, cz, res = w.result()
                    percent += 100 / (chunks[0]*chunks[1]*chunks[2])
                    print("\rGenerating probability map with {:d} threads... {:.2f}%".format(threads, percent), flush=True, end=" ")
                    cx *= chunk_size[0]
                    cy *= chunk_size[1]
                    cz *= chunk_size[2]
                    prob[cx:cx+chunk_size[0], cy:cy+chunk_size[1], cz:cz+chunk_size[2]] = res
                    workers.remove(w)
    # normalize distribution to be in range [0, 1]
    prob = (prob - np.amin(prob))/np.ptp(prob)
    print("Done. Total time: {:.2f}s".format(time.time() - t))
    return prob

# determine where clusters are located within the probability map
def find_clusters(prob, cutoff, age, img_size):
    print("Selecting cluster locations...", end="", flush=True)
    increment = 100 / img_size[0]
    percent = 0.00
    id = 1
    clusters = np.zeros(img_size).astype(int)
    locs = prob >= cutoff
    # iterate over image
    for x in range(0, img_size[0], 1):
        for y in range(0, img_size[1], 9):
            for z in range(0, img_size[2], 9):
                if locs[x,y,z]:
                    # calculate index ranges
                    x1 = np.clip(x-9, 0, img_size[0])
                    x2 = np.clip(x+10, 0, img_size[0])
                    y1 = np.clip(y-9, 0, img_size[1])
                    y2 = np.clip(y+10, 0, img_size[1])
                    z1 = np.clip(z-9, 0, img_size[2])
                    z2 = np.clip(z+10, 0, img_size[2])
                    # get space views
                    locview = locs[x1:x2, y1:y2, z1:z2]
                    clusview = clusters[x1:x2, y1:y2, z1:z2]
                    existing = clusview[clusview > 0]
                    if existing.size > 0:
                        clusview[locview] = np.random.choice(existing, clusview[locview].shape)
                    else:
                        clusview[locview] = id
                        id += 1
        percent += increment
        print("\rSelecting cluster locations... {:3.2f}%".format(percent), end="", flush=True)
        # util.write_cluster_image(clusters, img_size, "output/clusters_{:d}".format(x))
        # input()

    print("\rSelecting cluster locations... 100.0% Done")
    # generate ages for each cluster
    ages = np.zeros(id).astype(np.uint64)
    for i in range(id):
        ages[i] = f.new_age(0, age)
    return clusters, ages

# generate stars in space
def generate_stars(prob, clusters, count, img_size):
    print("Generating", count, "stars...", flush=True)
    # determine number of stars per spectral class using the Initial Mass Function
    # This is before age is taken into account
    classes = {}
    for c in "OBAFGKM":
        classes[c] = f.imf(f.new_mass(c))
    # calculate ratios and thus total number of stars of a given type
    total_imf = sum(classes.values())
    for c in "OBAFGKM":
        classes[c] = int(np.ceil(classes[c] / total_imf * count))
        print(c, "type stars:", classes[c])

    # generate stars
    stars = []
    while len(stars) < count:
        # randomly generate x, y, and z positions within the dimensions of the probability map
        pos = (np.random.randint(img_size[0]),
               np.random.randint(img_size[1]),
               np.random.randint(img_size[2]))
        # Get a random value in the range 0-1. If less than the probability at the star's location, save it.
        if np.random.random() < prob[pos]:
            avail = "OBAFGKM"
            # determine star's spectral class
            c = random.choice(avail)
            while(classes[c] == 0):
                avail.replace(c, "")
                c = random.choice(avail)
            classes[c] -= 1
            # save star.
            s = Star(pos, c, f.new_mass(c), clusters[pos])
            stars.append(s)
        print("\r{:.2f}%".format(len(stars) / count * 100), end=" ")
    print("Done")
    return stars

# age each star
def age_stars(stars, cluster_ages, universe):
    print("Aging stars... ", end="", flush=True)
    percent = 0
    for star in stars:
        # if star is in a cluster, use the cluster's age
        if star.cluster > 0:
            star.age = cluster_ages[star.cluster-1]
        # otherwise, get a new age
        else:
            star.age = f.new_age(0, universe)
        # determine the phase of life a star is in
        new_lum, new_temp = f.stage(star.mass, star.age)
        if new_lum != 0 or new_temp != 0:
            star._lum = 10**new_lum
            star._temp = 10**new_temp
    print("Done")
    return stars
