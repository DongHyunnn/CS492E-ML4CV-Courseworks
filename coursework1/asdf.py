
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

mat_file_name = "face.mat"

mat_content = sio.loadmat(mat_file_name)

print(mat_content) # Let's see the content... 