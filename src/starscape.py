'''
Randomly generates a starscape.
'''

from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import sys
import time
from opensimplex import OpenSimplex

import utility as util

# Real life constants with arbitrary values. Values represent what I felt was the best
# fit in the context of this simulation
L_sun = 1.0
R_sun = 1.0

img_size = (32, 1024, 1024)
feature_size = (8.0, 128.0, 128.0)
chunk_size = (32, 32, 32)

class Star:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.cluster = 0
    def pos(self):
        return (self.x, self.y, self.z)

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

def probability_map():
    print("Generating probability map (this will take a while)... 0.00%", flush=True, end=" ")
    # initialize data structures
    t = time.time()
    workers = []
    seed = np.random.randint(9223372036854775807)
    chunks = (np.asarray(img_size) // np.asarray(chunk_size)).astype(int)
    prob = np.zeros(img_size)
    # generate chunks in parallel
    with ThreadPoolExecutor(max_workers=6) as executor:
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
                    percent += 100 * ((cx*chunks[1]*chunks[2]) + (cy*chunks[2]) + cz) / (chunks[0]*chunks[1]*chunks[2])
                    print("\rGenerating probability map (this will take a while)... {:.2f}%".format(percent), flush=True, end=" ")
                    cx *= chunk_size[0]
                    cy *= chunk_size[1]
                    cz *= chunk_size[2]
                    prob[cx:cx+chunk_size[0], cy:cy+chunk_size[1], cz:cz+chunk_size[2]] = res
                    util.write_gs_img(prob, img_size, 'distribution', distance=True)
                    workers.remove(w)
    # normalize distribution to be in range [0, 1]
    prob = (prob - np.amin(prob))/np.ptp(prob)
    print("Done. Total time: {:.2f}s".format(time.time() - t))
    return prob

def find_clusters(prob, cutoff):
    print("Selecting cluster locations...", end=" ", flush=True)
    # determine location of clusters
    locs = prob > cutoff
    # initialize new image for testing
    clusters = np.zeros(img_size)
    id = 1
    # iterate over image
    for x in range(img_size[0]):
        for y in range(img_size[1]):
            for z in range(img_size[2]):
                # found a cluster. Label it with the current ID, or the next ID, if necessary
                if locs[x,y,z]:
                    locs[x,y,z] = False

                    # generate the indices of surrounding points in a 5-unit radius
                    neighbors = []
                    for offx in range(-5, 6):
                        for offy in range(-5, 6):
                            for offz in range(-5, 6):
                                index = (x+offx, y+offy, z+offz)
                                # for some reason tuple comparison was failing me here.
                                # index < img_size refused to work properly
                                if index[0] >= 0 and index[1] >= 0 and index[2] >= 0 and index[0] < img_size[0] and index[1] < img_size[1] and index[2] < img_size[2]:
                                    neighbors.append(index)
                    # search neighbors for an existing cluster
                    for point in neighbors:
                        if clusters[point] != 0:
                            # set current point's cluster as the neighbor's cluster
                            clusters[x,y,z] = clusters[point]
                            break

                    # if no existing clusters were found, set it to a new value
                    if clusters[x,y,z] == 0:
                        clusters[x,y,z] = id
                        id += 1
                    print("\rSelecting cluster locations... {:.2f}%".format(100 * ((x*img_size[1]*img_size[2]) + (y*img_size[1]) + z) / (img_size[0]*img_size[1]*img_size[2])), end=" ", flush=True)

    print("\rSelecting cluster locations... {:.2f}% Done".format(100.0))
    return clusters

# step 2a. Generates stars in random locations within the probability map.
# 'count' represents the number of stars to generate.
def generate_stars(prob, clusters, count=5000):
    print("Generating star locations...", end=" ", flush=True)
    stars = set()
    while len(stars) < count:
        # randomly generate x, y, and z positions within the dimensions of the probability map
        star = Star(np.random.randint(img_size[0]),
                    np.random.randint(img_size[1]),
                    np.random.randint(img_size[2]))
        # randomly decide if a star forms at that location. Get a random value in the range 0-1.
        # If less than the probability at the star's location, save it.
        if np.random.random() < prob[star.pos()]:
            star.cluster = clusters[star.pos()]
            stars.add(star)
    print("Done")
    return stars

# step 2b. Place stars in their own space. 'stars' is the set of points generated in star_locations
def place_stars(stars):
    print("Placing stars in space...", end=" ", flush=True)
    space = np.zeros(img_size)
    for star in stars:
        val = 1.0
        if star.cluster != 0:
            val = 2.0
        n = tuple(np.asarray(star.pos()) + np.asarray((0,1,0)))
        s = tuple(np.asarray(star.pos()) + np.asarray((0,-1,0)))
        e = tuple(np.asarray(star.pos()) + np.asarray((0,0,1)))
        w = tuple(np.asarray(star.pos()) + np.asarray((0,0,-1)))
        space[star.pos()] = val
        if n[1] < space.shape[1]:
            space[n] = val
        if s[1] > 0:
            space[s] = val
        if e[2] < space.shape[2]:
            space[e] = val
        if w[2] > 0:
            space[w] = val
    print("Done")
    return space

if __name__ == '__main__':
    # np.random.seed(0)
    prob = np.array([])
    if len(sys.argv) > 1:
        prob = probability_map()
        prob.tofile("data.raw")
    else:
        prob = np.reshape(np.fromfile("data.raw"), img_size)
    prob = np.power(prob, 3)
    util.write_gs_img(prob, img_size, 'distribution', distance=True)
    clusters = find_clusters(prob, 0.7)
    util.write_gs_img(clusters, img_size, "clusters")
    space = place_stars(generate_stars(prob, clusters, 10000))
    util.write_gs_img(space, img_size, 'stars', distance=True)
