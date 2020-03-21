'''
Formulae. Inverse square law, H-R diagram interpretation, etc.
'''
import numpy as np

# inverse square law of brightness
# Assumes 'dist' is nonzero
def inv_sq(lum, dist):
    return lum / (4.0 * np.pi * np.square(dist+1))
