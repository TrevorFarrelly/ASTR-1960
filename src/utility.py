'''
Utility. Reading/Writing data, taking user input, etc. No astronomical processing
happens here.
'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import formula as f

# Write distribution map to image. Expects 3D numpy space
def write_dist_img(data, img_size, name):
    print("Writing {:s}.png to disk... ".format(name), end="", flush=True)
    # normalize data to range [0-1]
    data = (data - np.amin(data))/np.ptp(data)
    # copy data to image array
    img = np.zeros(img_size[1:])
    for i in range(data.shape[0]):
        img += f.inv_sq(data[i], i, 2)
    # normalize image to range [0-255]
    img = ((img - np.amin(img))/np.ptp(img)*255).astype(int)
    plt.imsave(name+".png", img, cmap="gray")
    print("Done")

# Write clusters to image. Expects 3D numpy space of integers
def write_cluster_image(data, img_size, name):
    print("Writing {:s}.png to disk... ".format(name), end="", flush=True)
    # generate random colors for each cluster
    colors = np.random.randint(32, 255, (np.amax(data),3)).astype(np.uint8)
    # copy data to image array
    img = np.zeros((img_size[1], img_size[2], 3)).astype(np.uint8)
    for i in range(data.shape[0]):
        # get free pixels that are in clusters
        pixels = data[i] > 0
        # set pixel color according to the cluster-color map
        img[pixels] = colors[data[i][pixels]-1]
    # write image to disk
    plt.imsave(name+".png", img)
    print("Done")

# write stars to image
def write_star_image(stars, img_size, name, distance=2):
    print("Writing {:s}.png to disk (exposure {:d})... ".format(name, distance), end="", flush=True)

    # initialize image
    img = np.zeros((img_size[1], img_size[2], 3)).astype(np.uint8)
    i = 0
    # calculate distance modifiers
    mods = np.zeros(len(stars)).astype(np.float64)
    for star in stars:
        mods[i] = f.inv_sq(star.img_lum(), star.pos()[0], distance)
        i += 1
    # normalize to range 0-1
    mods = (mods - np.amin(mods))/np.ptp(mods)

    # iterate over all stars
    for star in stars:
        # modify color by distance and luminosity
        val = (np.array(star.color()) * mods[stars.index(star)]).astype(np.uint8)
        # get all surrounding points
        n = tuple(np.asarray(star.pos()) + np.asarray((0,1,0)))
        s = tuple(np.asarray(star.pos()) + np.asarray((0,-1,0)))
        e = tuple(np.asarray(star.pos()) + np.asarray((0,0,1)))
        w = tuple(np.asarray(star.pos()) + np.asarray((0,0,-1)))
        # set all surrounding points to that color if they are in the image
        img[star.pos()[1:3]] = val
        if n[1] < img_size[1] and np.all(np.less(img[n[1:3]], val)):
            img[n[1:3]] = val
        if s[1] > 0 and np.any(np.less(img[s[1:3]], val)):
            img[s[1:3]] = val
        if e[2] < img_size[2] and np.any(np.less(img[e[1:3]], val)):
            img[e[1:3]] = val
        if w[2] > 0 and np.any(np.less(img[w[1:3]], val)):
            img[w[1:3]] = val
    plt.imsave(name+".png", img)
    print("Done")

def write_HR_diagram(stars, name):
    print("Writing HR diagram to disk... ", end="", flush=True)
    # initialize plot
    plt.rcParams["figure.figsize"] = [9,12]
    mintemp = np.Inf
    maxtemp = 0
    minlum = np.Inf
    maxlum = 0
    i = 0
    # initialize data arrays
    x = np.zeros((len(stars)+1))
    y = np.zeros((len(stars)+1))
    c = np.zeros((len(stars)+1, 3))
    # iterate over stars
    for star in stars:
        # determine approximate absolute temperature
        x[i] = np.log10(star.temp() * 1000)
        y[i] = star.lum()
        c[i] = np.array(star.color()) / 255
        if x[i] < mintemp:
            mintemp = x[i]
        if x[i] > maxtemp:
            maxtemp = x[i]
        if y[i] < minlum:
            minlum = y[i]
        if y[i] > maxlum:
            maxlum = y[i]
        i += 1
    # add the sun as a dark blue point for reference
    x[len(stars)] = np.log10(5778)
    y[len(stars)] = 1
    c[len(stars)] = np.array((255,255,0)) / 255
    # add data to plot
    plt.scatter(x, y, s=1, color=c)
    # modify figure settings
    plt.xlabel('Surface Temperature (log(K))')
    plt.gca().set_xlim(maxtemp, mintemp)
    plt.ylabel('Luminosity (L_sun)')
    plt.gca().set_ylim(minlum, maxlum)
    plt.yscale('log')
    plt.gca().set_facecolor('#282B32')
    plt.savefig(name+".png", bbox_inches='tight')
    print("Done")
