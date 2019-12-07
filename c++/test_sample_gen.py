import numpy as np
# generate random test sample for comparing cpp and py
data = np.random.uniform(low=[-1.0]*78, high=[1.0]*78)
np.savetxt('test_sample.txt', data, delimiter='\n', fmt='%f')
