'''
Formulae
'''
import numpy as np


# Randomly generate mass for a given class type.
# Value will be in range cm +/- 0.5(cm) for each class.
def new_mass(c):
    cm = {"O":60, "B":18, "A":3.2, "F":1.7, "G":1.1, "K":0.65, "M":0.3}
    return cm[c]+np.random.uniform(-0.5*cm[c], 0.5*cm[c])

# generate a random age, in the range 0-13*10^10. Allow passing in a value that
# limits the return value, ensuring that nothing is older than the universe
def new_age(start, universe):
    universe = (universe * 10**9)
    if universe - start <= 0:
        return 0
    return int(np.random.beta(3.33,6.66) * (universe - start))

# Kroupa's 2001 model of the Intitial Mass Function.
def imf(mass):
    if mass < 0.08:
        return mass**(-1*0.3)
    elif mass < 0.5:
        return mass**(-1*1.3)
    else:
        return mass**(-1*2.3)

# calculate luminosity from mass, relative to the luminosity of the sun. page 139,
# https://books.google.com/books?id=r1dNzr8viRYC&pg=PA138
def lum(mass):
    val = 0
    if mass < 0.43:
        val = 0.23 * mass**2.3
    elif mass < 2:
        val = mass**4
    elif mass < 55:
        val = 1.4 * mass**3.5
    else:
        val = 32000 * mass
    return val + np.random.uniform(-0.1*val, 0.1*val) + 2 * np.random.uniform(-0.05*val, 0.05*val)

# approximate temperature from mass, based on the spectral standard stars for each
# spectral type. https://en.wikipedia.org/wiki/Stellar_classification#Spectral_types
def temp(mass):
    val = 6.21 * np.power(mass, 0.533)
    return val + np.random.uniform(-0.1*val, 0.1*val) + 2 * np.random.uniform(-0.05*val, 0.05*val)

# calculate color from temperature. Sampled from data at
# https://academo.org/demos/colour-temperature-relationship/
def color(temp, offset=1):
    # RGB values for temperatures between 1500K-15000K, manually tweaked by adding
    # duplicate colors. Since even the coolest stars are hotter than 1500K,
    # we artificially increase this range by decreasing
    # the star's temperature by a constant offset.
    colors = [(255, 140, 50), (255,143,55), (255,158,79),
    (255, 177, 110), (255, 206, 166), (255,215,182),
    (255, 228, 206), (255, 255, 245),
    (255, 255, 255), (255, 255, 255),(255, 255, 255),
    (255, 255, 255), (243, 242, 255),(221, 231, 255),
    (210, 223, 255), (196, 214, 255), (191, 211, 255),
    (202, 218, 255), (185, 207, 255), (179, 202, 255),
    (171, 195, 255), (151, 175, 255)]
    # select closest valid color from the list
    return colors[np.clip(temp-offset, 0, len(colors)-1).astype(int)]

# inverse square law of brightness.
def inv_sq(lum, dist, off):
    return lum / (4.0 * np.pi * np.square(dist+off))

# determine the stage of life a star is in, based on its mass and age
# data approximated from slides 16 and 17 of https://www.astro.caltech.edu/~george/ay20/Ay20-Lec9x.pdf
def stage(mass, age):
    T = []
    L = []
    y = []
    if mass < 0.875:
        return 0,0
    elif 0.875 <= mass and mass < 1.125:
        T = [np.log10(temp(mass)), 0.76, 0.79, 0.785, 0.74, 0.695, 0.55]
        L = [np.log10(lum(mass)), 0, 0.2, 0.3, 0.48, 0.45, 2.6]
        Y = [0, mass**-2.5*10**10, 7*10**9, 2*10**9, 1.2*10**9, 1.57*10**8, 2*10**9]
    elif 1.125 <= mass and mass < 1.375:
        T = [np.log10(temp(mass)), 0.83, 0.815, 0.85, 0.79, 0.69, 0.55]
        L = [np.log10(lum(mass)), 0.35, 0.55, 0.6, 0.8, 0.75, 2.65]
        Y = [0, mass**-2.5*10**10, 2.803*10**9, 1.824*10**9, 1.045*10**9, 1.463*10**8, 5*10**8]
    elif 1.375 <= mass and mass < 1.875:
        T = [np.log10(temp(mass)), 0.92, 0.86, 0.905, 0.85, 0.69, 0.55]
        L = [np.log10(lum(mass)), 0.7, 0.85, 0.95, 1.15, 0.95, 2.7]
        Y = [0, mass**-2.5*10**10, 1.553*10**9, 8.1*10**7, 3.49*10**8, 1.049*10**8, 3*10**8]
    elif 1.875 <= mass and mass < 2.625:
        T = [np.log10(temp(mass)), 1.05, 0.95, 1.025, 0.98, 0.69, 0.7]
        L = [np.log10(lum(mass)), 1.5, 1.7, 1.75, 1.8, 1.5, 2.8]
        Y = [0, mass**-2.5*10**10, 4.802*10**8, 1.647*10**7, 3.696*10**7, 1.31*10**7, 3.829*10**7]
    elif 2.625 <= mass and mass < 4:
        T = [np.log10(temp(mass)), 1.45, 1.06, 1.1, 1.05, 0.69, 0.61, 0.67, 0.75, 0.64]
        L = [np.log10(lum(mass)), 1.99, 2.15, 2.2, 2.4, 1.99, 2.45, 2.15, 2.4, 2.4]
        Y = [0, mass**-2.5*10**10, 2.212*10**8, 1.042*10**7, 1.033*10**7, 4.505*10**6, 4.238*10**6, 2.51*10**7, 4.08*10**7, 6*10**6]
    elif 4 <= mass and mass < 7:
        T = [np.log10(temp(mass)), 1.285, 1.2, 1.24, 1.18, 0.66, 0.61, 0.65, 0.75, 0.91, 0.69]
        L = [np.log10(lum(mass)), 2.8, 3, 3.1, 3.17, 2.9, 3.15, 3.05, 3.17, 3.4, 3.38]
        Y = [0, mass**-2.5*10**10, 6.547*10**7, 2.173*10**6, 1.372*10**6, 7.532*10**5, 4.857*10**5, 6.05*10**6, 1.02*10**6, 9*10**6, 9.3*10**5]
    elif 7 <= mass and mass < 12:
        T = [np.log10(temp(mass)), 1.42, 1.335, 1.38, 1.28, 0.645, 0.6, 0.61, 1.05, 1.13, 0.97]
        L = [np.log10(lum(mass)), 3.6, 3.9, 3.95, 4, 3.8, 4.02, 3.97, 3.97, 4.25, 4.3]
        Y = [0, mass**-2.5*10**10, 2.144*10**7, 6.053*10**5, 9.133*10**4, 1.477*10**5, 6.552*10**4, 4.9*10**5, 9.5*10**4, 3.28*10**6, 1.55*10**5]
    elif 12 <= mass and mass < 25:
        T = [np.log10(temp(mass)), 1.515, 1.42, 1.48, 1.26, 1.2, 1.11, 0.98, 0.61]
        L = [np.log10(lum(mass)), 4.35, 4.6, 4.65, 4.75, 4.8, 4.9, 4.92, 4.9]
        Y = [0, mass**-2.5*10**10, 1.01*10**7, 2.270*10**5, 7.55*10**4, 7.17*10**5, 6.2*10**5, 1.9*10**5, 3.5*10**4]
    # turn all dying high-mass stars into neutron stars for ease of computation
    else:
        if age < mass**-2.5*10**10:
            return 0,0
        return np.log10(lum(new_mass("M"))), np.log10(temp(new_mass("O")))

    # iterate over all stages of life for this star
    for i in range(1, len(T)-1):
        # if we found the stage this star lives in, determine where it is within the stage
        if age < Y[i]:
            Y_delta = (age - Y[i-1]) / (Y[i] - Y[i-1])
            new_T = T[i] + ((T[i] - T[i-1]) * Y_delta)
            new_L = L[i] + ((L[i] - L[i-1]) * Y_delta)
            return new_L + np.random.uniform(-0.07*new_L, 0.07*new_L), new_T + np.random.uniform(-0.07*new_T, 0.07*new_T)
        age -= Y[i]
    # if we exit the loop without finding a solution, this star is a white dwarf
    return np.log10(lum(new_mass("M"))), np.log10(temp(mass))
