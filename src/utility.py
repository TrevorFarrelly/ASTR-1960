'''
Utility. Reading/Writing data, taking user input, etc. No astronomical processing
happens here.
'''
import cv2
import numpy as np
import formula as f

# Draw circle

# Write data to a greyscale image. Expects a 3D numpy array.
def write_gs_img(data, img_size, name, distance=False):
    # flatten image using the inverse square law.
    img = np.zeros(img_size[1:])
    for i in range(np.size(data,0)):
        if distance:
            img += f.inv_sq(data[i], i+1)
        else:
            img = np.maximum(img, data[i])

    # normalize data to range [0-255]
    img = ((img - np.amin(img))/np.ptp(img)*255).astype(int)
    # write image to disk
    cv2.imwrite(name+'.png', img)
