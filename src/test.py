import numpy as np

testarr = np.zeros((32,64,64))
arr1 = np.ones((32,32,32))

print(testarr, "\n=======================================\n")
testarr[0:32, 0:32, 0:32] = arr1
print(testarr, "\n=======================================\n")
testarr[0:32, 32:64, 32:64] = arr1
print(testarr, "\n=======================================\n")
